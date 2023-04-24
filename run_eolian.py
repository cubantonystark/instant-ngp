import os, glob, sys, shutil, argparse, cv2, screeninfo, platform, logging
from halo import Halo
from tqdm import tqdm
from PIL import Image
import torch
from carvekit.api.high import HiInterface

level    = logging.INFO
format   = '%(message)s'
handlers = [logging.StreamHandler()]
logging.basicConfig(level = level, format='%(asctime)s \033[1;34;40m%(levelname)-8s \033[1;37;40m%(message)s',datefmt='%H:%M:%S', handlers = handlers)

class processor():
    
    def main(self):
        
        #Declare some global variables
        global mode, display_mode, heigth, width, ps, start_time, source_dir, work_dir
        
        #Get the OS
        _os = platform.platform()
        
        #The slash will be based on the OS - Thanks winblows for that.

        if 'Windows' in _os:
            ps = '\\'
        
        else:
            ps = '/'
        
        #Check how many monitors there are
        
        monitors = screeninfo.get_monitors()
        
        if len(monitors) == 1:
            display_mode = ' '
            
        #If there is more than one diasplay detected, we will open an additional screen as a briefing screen
        else:
            display_mode = ' --second_window'
            
        for monitor in monitors:

            if monitor.is_primary == True: 
                width = monitor.width
                heigth = monitor.height
        
        #cleanup images folder on root if there is one
        if os.path.exists('images'):  
            shutil.rmtree('images')
        if os.path.exists('nerf'):  
            shutil.rmtree('nerf')    
        if os.path.exists('transforms.json'):  
            os.unlink('transforms.json')   
        if os.path.exists('snapshot.ingp'):  
            os.unlink('snapshot.ingp')        
        
        #parse passed arguments        
        parser = argparse.ArgumentParser()

        parser.add_argument(type = str, dest = 'src_folder', help = 'Target location where to look for images. Default: script folder')
        parser.add_argument('-m', '--mode', dest = 'mode', type = str, help = '\'scene\' or \'object\'. Either render a full scene or extract object from scene and render. Default: \'scene\'', required = False, default = 'scene')

        args = parser.parse_args()
        
        #if no folder was passed, assume the root folder as the one with images
        
        if args.src_folder is None:
            args.src_folder = os.getcwd()
            
        source_dir = args.src_folder+ps
        work_dir = os.getcwd()+ps
        mode = args.mode
        
        #Check if its a video file
        self.extract_images_from_video()
        
        #Check if there is already a checkpoint saved
        self.direct_render_checkpoint()
        
        #if there is no checkpoint, check if there is a training json file saved
        self.direct_render_transforms()
        
        #if there are no video, checkpoints or training files, transfer the images folder here.
        
        try:
            
            shutil.copytree(source_dir+'images', work_dir+'images')
            
        except:
            
            logging.info('No compatible image format found')
            sys.exit()
        
        #then continue and check for png extensions
        self.check_extension_and_convert_if_needed()
        
    def check_extension_and_convert_if_needed(self):
        
        files = [f for f in glob.glob(work_dir+'images'+ps+'*.*')]
        
        if len(files) == 0:
            logging.info('No compatible image format found')
            sys.exit()            
            
        filename, extension = files[0].split('.')

        if (extension == 'png') or (extension == 'PNG'):
            logging.info('Image type: PNG (Portable Network Graphics)')         
            self.resizer(files)

        elif (extension == 'jpg') or (extension == 'JPG'):
            
            logging.info('Image type: JPG (Joint Photographic Experts Group')
            logging.info('Runnig conversion')
            for file in tqdm(files):
                file_to_png = file
                img = Image.open(file_to_png)
                file_to_png = file_to_png.replace('.JPG', '')
                file_to_png = file_to_png.replace('.jpg', '')
                img.save(file_to_png+'.png'.lower())
                img.close()
                os.remove(file)

            files = [f for f in glob.glob('images'+ps+'*.png')]
    
            for file in files:
                file_to_rename = file.replace('.png.png', '.png')
                os.rename(file, file_to_rename)
            
            files = [f for f in glob.glob('images'+ps+'*.png')]
            self.resizer(files)

        else:
            logging.info('No compatible image format found')
            sys.exit()    
        
    def resizer(self, files):
        
        image = Image.open(files[0])
        img_heigth, img_width = image.size
        image.close()
        if (img_heigth != 1920 or img_width != 1080):
            logging.info('Adjusting resolution')
            for file in tqdm(files):
                
                #Resize to 1024 x 786 for memory purposes
                image = Image.open(file)
                new_image = image.resize((1920, 1080))
                new_image.save(file)

        if mode == 'scene':
            spinner = Halo(text='Computing image transforms', spinner='dots')
            spinner.start()            
            command = 'python ./scripts/colmap2nerf.py --colmap_matcher exhaustive --run_colmap --aabb_scale 16 --images images > log.log'
            os.system(command) 
            spinner.stop()
            
            if os.path.exists('colmap_sparse'):
                shutil.rmtree('colmap_sparse') 
            if os.path.exists('colmap_text'):
                shutil.rmtree('colmap_text')
            if os.path.exists('colmap.db'):
                os.unlink('colmap.db')            
            
            command = 'python ./scripts/run.py --scene '+work_dir+' --train --gui'+display_mode+' --height '+str(heigth)+' --width '+str(width)+' --save_snapshot snapshot.ingp' 
            os.system(command)
            
            os.mkdir('nerf')   
            shutil.copy2('transforms.json', 'nerf')
            shutil.copy2('snapshot.ingp', 'nerf')
            shutil.copytree('images', 'nerf'+ps+'images')
            shutil.copytree('nerf', source_dir+ps+'nerf'+ps)
            shutil.rmtree('nerf')
            shutil.rmtree('images')
            sys.exit()
            
        else:
            
            self.remove_bg()
            
    def remove_bg(self):
        
        if os.path.exists('tmp'):
            shutil.rmtree('tmp')
            os.mkdir('tmp')
            os.mkdir('tmp'+ps+'images')            
            
        else:
            os.mkdir('tmp')
            os.mkdir('tmp'+ps+'images')  
            
        if torch.cuda.is_available():
            processor = 'GPU'
            
        else:
            processor = 'CPU'  
            
        logging.info('Running process on '+processor) 
        
        #Remove Background
 
        interface = HiInterface(batch_size_seg=5, batch_size_matting=1,
                                       device='cuda' if torch.cuda.is_available() else 'cpu',
                                       seg_mask_size=320, matting_mask_size=2048)
        
        files = [f for f in glob.glob('images'+ps+'*.png')]
        for image in tqdm(files, desc = 'Removing background'):
            images_without_background = interface([image])      
            wo_bg = images_without_background[0]
            wo_bg.save(work_dir+'tmp'+ps+image.replace(work_dir, ''))
            
        
        #Remove alphas
        files = [f for f in glob.glob('tmp'+ps+'images'+ps+'*.png')]
        
        for image in files:
            filename = image
            image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)    
            #make mask of where the transparent bits are
            trans_mask = image[:,:,3] == 0
            #replace areas of transparency with white and not transparent
            image[trans_mask] = [255, 255, 255, 0]
            #new image without alpha channel...
            #new_img = cv2.cvtColor(image)
            cv2.imwrite(filename, image)  
        
        shutil.rmtree('images')  
        os.mkdir('images')
        shutil.copytree('tmp'+ps+'images', 'images')
        shutil.rmtree('tmp')
        
        spinner = Halo(text='Computing image transforms', spinner='dots')
        spinner.start()            
        command = 'python ./scripts/colmap2nerf.py --colmap_matcher exhaustive --run_colmap --aabb_scale 16 --images images > log.log'
        os.system(command)
        spinner.stop()  
        
        if os.path.exists('colmap_sparse'):
            shutil.rmtree('colmap_sparse') 
        if os.path.exists('colmap_text'):
            shutil.rmtree('colmap_text')
        if os.path.exists('colmap.db'):
            os.unlink('colmap.db')
        
        command = 'python ./scripts/run.py --scene '+work_dir+' --train --gui --n_steps 35000'+display_mode+' --height '+str(heigth)+' --width '+str(width)+' --save_snapshot snapshot.ingp'
        os.system(command)
        
        os.mkdir('nerf')   
        shutil.copy2('transforms.json', 'nerf')
        shutil.copy2('snapshot.ingp', 'nerf')
        shutil.copytree('images', 'nerf'+ps+'images')
        shutil.copytree('nerf', source_dir+ps+'nerf'+ps)
        shutil.rmtree('nerf')
        shutil.rmtree('images')
        sys.exit()
        
    def extract_images_from_video(self):
        
        file = glob.glob(source_dir+'*.mp4')
        
        try:
            
            if os.path.exists(source_dir+'images'+ps) and len(file[0] > 0):
                pass 
            
            else:
                
                os.mkdir(source_dir+ps+'images'+ps)
                logging.info('Extracting frames from video')
                command = 'ffmpeg -i '+file[0]+' -vf fps=1/0.50 '+source_dir+'images'+ps+'%04d.png -loglevel quiet'
                os.system(command)                
                
        except:
            
            pass

        return
        
    def direct_render_transforms(self):
        
        files = glob.glob(source_dir+ps+'transforms.json')
        
        if len(files) == 0:
            return
        
        else:
            command = 'python ./scripts/run.py --scene '+source_dir+' --train --gui --n_steps 35000'+display_mode+' --height '+str(heigth)+' --width '+str(width)+' --save_snapshot '+source_dir+ps+'snapshot.ingp' 
            os.system(command)
            sys.exit()
            
    def direct_render_checkpoint(self):
        
        files = glob.glob(source_dir+'*.ingp')
        
        if len(files) == 0:
            return
        
        else:
            command = 'python ./scripts/run.py --load_snapshot '+files[0]+' --train --gui '+display_mode+' --height '+str(heigth)+' --width '+str(width)+' --save_snapshot '+source_dir+ps+'snapshot.ingp'
            os.system(command)
            sys.exit()        

if __name__ == '__main__':

    processor().main()