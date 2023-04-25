import numpy as np
import cv2
import math

#variable to set the debugge mode to off/on
debugge_mode=1

#variable set to 1 if nav direction is toward a rock
is_rock_nav_direction=0

#variable set to 1 if nav direction is toward ground
is_ground_nav_direction=0

 
    
#*********************************************************    
#begin editing 
#function to calculate the source and destination point for the image wraping
def calc_source_and_destination(img):
    dst=3
    bottom_offset=5
    croprd_image=img[90:150,:]
    #plt.imshow(croprd_image)
    gry=cv2.cvtColor(croprd_image,cv2.COLOR_BGR2GRAY)
    corners=cv2.goodFeaturesToTrack(gry,4,0.5,10)
    corners=np.int0(corners)
    x=[]
    y=[]
    for i in corners:
        b,n=i.ravel()
        print(b,n)
        x.append(b)
        y.append(n)

    source =np.float32([[x[2],y[2]+60],
                      [x[3],y[3]+60],
                      [x[0],y[0]+60],
                       [x[1],y[1]+60]])         
    destination =np.float32([[img.shape[1]/2-dst,img.shape[0]-bottom_offset],
                             [img.shape[1]/2+dst,img.shape[0]-bottom_offset],
                             [img.shape[1]/2+dst,img.shape[0]-2*dst-bottom_offset],
                             [img.shape[1]/2-dst,img.shape[0]-2*dst-bottom_offset]])
    
    return source, destination

#funciton to identify the ground pixels
#output: 
#ground pixels in white
#other in black
def ground_thres_new(image,mask_obstacles, mask_rock ):
    mask_obstacles_or_rock =cv2.bitwise_or(mask_obstacles,mask_rock,mask=None) 
    mask_ground =np.invert(mask_obstacles_or_rock)
    return mask_ground

def ground_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

def ground_thres(image, lower_limit=(0, 0, 150),upper_limit=(153, 131, 255)):
    bgr_ground = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    hsv_ground = cv2.cvtColor(bgr_ground,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_ground,lower_limit,upper_limit)
    
    kernel = np.ones((7, 7), np.uint8)  
    #close operation
    detected_obstacle = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    #open operation
    detected_obstacle = cv2.morphologyEx(detected_obstacle, cv2.MORPH_CLOSE, kernel)  
    
    detected_obstacles_image = cv2.cvtColor(detected_obstacle,cv2.COLOR_BGR2RGB)
    
    detected_obstacles_image[detected_obstacles_image>0]=255
    detected_obstacles_image[detected_obstacles_image==0]=0
    
    return detected_obstacles_image[:,:,0]   

def ground_thres_last_edited(image,lower=(0, 0, 88),upper=(179, 255, 255)):
   # bgr_obstecals = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    hsv_obsteclas = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_obsteclas,lower,upper)  
    # define the kernal
    kernel = np.ones((5, 5), np.uint8)  
    #close operation
    detected_obstacle = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    #open operation
    detected_obstacle = cv2.morphologyEx(detected_obstacle, cv2.MORPH_CLOSE, kernel)
    detected_obstacle_rgb = cv2.cvtColor(detected_obstacle,cv2.COLOR_BGR2RGB)
    #detected_obstacles_image =cv2.cvtColor(detected_obstacle,cv2.COLOR_RGB2GRAY)
    #invert the white and black pixels
    #mask_obstacles_or_rock =cv2.bitwise_and(mask_obstacles,mask_rock,mask=None)
    #detected_obstacles_image =np.invert(detected_obstacle_rgb)
    #detected_obstacles_image =cv2.cvtColor(detected_obstacles_image,cv2.COLOR_BGR2GRAY) 
    return  detected_obstacle_rgb[:,:,0]   

#funciton to identify the rock pixels
#output: 
#rock pixels in white
#other in black


def detect_rock(image,lower=(91, 85, 84),upper=(102, 255, 255)):
    hsv_rock = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_rock,lower,upper)  
    #res = cv2.bitwise_and(image, image, mask=mask)
    detected_rock_image = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
    detected_rock_image[detected_rock_image>0]=255
    detected_rock_image[detected_rock_image==0]=0
    #detected_rock_image = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
    return detected_rock_image[:,:,0]




#funciton to identify the obstecals pixels
#output: 
#obstecals pixels in white
#other in black

def detect_obstecals(image,lower=(0, 0, 88),upper=(179, 255, 255)):
   # bgr_obstecals = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    hsv_obsteclas = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_obsteclas,lower,upper)  
    # define the kernal
    kernel = np.ones((5, 5), np.uint8)  
    #close operation
    detected_obstacle = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    #open operation
    detected_obstacle = cv2.morphologyEx(detected_obstacle, cv2.MORPH_CLOSE, kernel)
    detected_obstacle_rgb = cv2.cvtColor(detected_obstacle,cv2.COLOR_BGR2RGB)
    #detected_obstacles_image =cv2.cvtColor(detected_obstacle,cv2.COLOR_RGB2GRAY)
    #invert the white and black pixels
    detected_obstacles_image =np.invert(detected_obstacle_rgb)
    #detected_obstacles_image =cv2.cvtColor(detected_obstacles_image,cv2.COLOR_RGB2GRAY) 
    detected_obstacles_image[detected_obstacles_image>0]=255
    detected_obstacles_image[detected_obstacles_image==0]=0
    
    return  detected_obstacles_image[:,:,0]   

def detect_obstecals_for_driving(image,lower=(0, 0, 88),upper=(179, 255, 255)):
   # bgr_obstecals = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    hsv_obsteclas = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_obsteclas,lower,upper)  
    # define the kernal
    kernel = np.ones((5, 5), np.uint8)  
    #close operation
    detected_obstacle = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    #open operation
    detected_obstacle = cv2.morphologyEx(detected_obstacle, cv2.MORPH_CLOSE, kernel)
    detected_obstacle_rgb = cv2.cvtColor(detected_obstacle,cv2.COLOR_BGR2RGB)
    #detected_obstacles_image =cv2.cvtColor(detected_obstacle,cv2.COLOR_RGB2GRAY)
    #invert the white and black pixels
    detected_obstacles_image =np.invert(detected_obstacle_rgb)
    #detected_obstacles_image =cv2.cvtColor(detected_obstacles_image,cv2.COLOR_RGB2GRAY) 
#     detected_obstacles_image[detected_obstacles_image>0]=255
#     detected_obstacles_image[detected_obstacles_image==0]=0
    
    return  detected_obstacles_image[:,:,0]   


#Define a function to detect if there obstacles at the:
#center of image
#left of the image
#right of the image

def is_obstacles_at_the_center(Image,shape_x,shape_y):
    cropped_Image=Image[:,0:int(shape_y/4)]
    
    if (cropped_Image.any(axis=-1).sum() >10) : #20 in case if noises exist
        return 1
    return 0

def is_obstacles_at_left(Image,shape_x,shape_y):
    cropped_Image=Image[:,int(shape_y/4):int(3*(shape_y/4))]
    if (cropped_Image.any(axis=-1).sum() >10) : #20 in case if noises exist
        return 1
    return 0

def is_obstacles_at_right(Image,shape_x,shape_y):
    cropped_Image=Image[:,int(3*(shape_y/4)):(shape_y)]
    if (cropped_Image.any(axis=-1).sum() >10) : #20 in case if noises exist
        return 1
    return 0

def is_rock_near(Image):
    if (Image.any(axis=-1).sum() >5) : #7 in case if noises exist
        return 1
    return 0



def prepare_image_for_show(img,text):
    
    #set a fixed size for all images 
    image_resize_width, image_resize_heigh =200,200
    
    img =cv2.resize(img,(image_resize_width,image_resize_heigh))
        
    padding_width, padding_height =int((image_resize_width+image_resize_width/5)), int((image_resize_heigh+image_resize_heigh/10))
        
    padding= np.ones((padding_width,padding_height,3), dtype=np.uint8)*125 #, dtype=np.unit8
        
    #replace the padding center with the original image
    padding_top , padding_left =20,5
    padding[padding_top:padding_top+image_resize_heigh, 0:image_resize_width]=img
        
    img1 =cv2.putText(padding.copy(),text,(int(0.25*padding_width),10),cv2.FONT_HERSHEY_COMPLEX,0.40,(255,0,0))

    return img1


def get_right_obstacle_frame(cropped_obstacle_image,shape_x,shape_y):
    cropped_Image=Image[:,int(3*(shape_y/4)):(shape_y)]
    return cropped_Image
    


                             
#end
#*********************************************************    

# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
                            
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result  
    return xpix_rotated, ypix_rotated

def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result  
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    
    return warped


# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
   
    
    cropped_image=Rover.img[95:145,14:300] 
    shape_x=cropped_image.shape[0] 
    shape_y =cropped_image.shape[1]  
    cropped_obstacle_image=detect_obstecals_for_driving(cropped_image)
    #set the variables
    Rover.obstacle_at_the_end_of_pixel=is_obstacles_at_the_center(cropped_obstacle_image,shape_x,shape_y)
    Rover.left_side_obstacle=is_obstacles_at_left(cropped_obstacle_image,shape_x,shape_y)
    Rover.right_side_obstacle=is_obstacles_at_right(cropped_obstacle_image,shape_x,shape_y)
    Rover.two_sides_obstacles =(Rover.left_side_obstacle & Rover.right_side_obstacle)  
    
  
    #cropped_rock=detect_rock(cropped_image)

               
    # 1) Define source and destination points for perspective transform 
    
    dst=20 #8/(9)/(10)
    bottom_offset=5
    croprd_image=Rover.img[90:150,:]
    source=np.float32([[14,140],
                      [300,140],
                      [200,95],
                      [120,95]])
    
    destination =np.float32([[Rover.img.shape[1]/2-dst,Rover.img.shape[0]-bottom_offset],
                             [Rover.img.shape[1]/2+dst,Rover.img.shape[0]-bottom_offset],
                             [Rover.img.shape[1]/2+dst,Rover.img.shape[0]-2*dst-bottom_offset],
                             [Rover.img.shape[1]/2-dst,Rover.img.shape[0]-2*dst-bottom_offset]])
    
    
   
    # 2) Apply perspective transform
    wraped_image=perspect_transform(Rover.img,source,destination)
    
    
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    #identify obstecals
    
    wraped_image_gray=wraped_image.copy()
    wraped_image_gray[wraped_image_gray>0]=255
    wraped_image_gray[wraped_image_gray==0]=0
    wraped_image_gray =cv2.cvtColor(wraped_image_gray,cv2.COLOR_RGB2GRAY)
 
    rock_image =cv2.bitwise_and(detect_rock(wraped_image),wraped_image_gray,mask=None)
    ground_image =cv2.bitwise_and(color_thresh(wraped_image),wraped_image_gray,mask=None)
    obstacles_image=cv2.bitwise_and(cv2.bitwise_not(ground_image),wraped_image_gray,mask=None)
    
    
    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
             
    #put the obstacles in the r channel
    Rover.vision_image[:,:,0]=obstacles_image  
    
    #put the rock in the g channel
    Rover.vision_image[:,:,1]=rock_image  
    
    #put the ground in the b channel
    Rover.vision_image[:,:,2]=ground_image  
    
     
    # 5) Convert map image pixel values to rover-centric coords
    x_rover_obstacles, y_rover_obstacles =rover_coords(obstacles_image)
    x_rover_rock, y_rover_rock =rover_coords(rock_image)
    x_rover_ground, y_rover_ground =rover_coords(ground_image)
        
        
        
    # 6) Convert rover-centric pixel values to world coordinates
    xpos =Rover.pos[0] # current Rover x position
    ypos =Rover.pos[1] # current Rover y position
    yaw =Rover.yaw
    world_size =Rover.worldmap.shape[0] #shape[0] to get the length of the worldmap
    
    scale =50 #24/30/(30) is the dist
    
    
    x_world_obstacles, y_world_obstecals = pix_to_world(x_rover_obstacles, y_rover_obstacles, xpos, ypos, yaw, world_size, scale)
    x_world_rock, y_world_rock =pix_to_world(x_rover_rock, y_rover_rock, xpos, ypos, yaw, world_size, scale)
    x_world_ground, y_world_ground =pix_to_world(x_rover_ground, y_rover_ground,xpos, ypos, yaw, world_size, scale)
    
    
        
    # 7) Update Rover worldmap (to be displayed on right side of screen)
        
    Rover.worldmap[y_world_obstecals, x_world_obstacles, 0] = 255 
    Rover.worldmap[y_world_rock, x_world_rock, 0] = 0   
 
    Rover.worldmap[y_world_rock, x_world_rock, 1] = 255   
    Rover.worldmap[y_world_ground, x_world_ground, 1] =0

    Rover.worldmap[y_world_ground, x_world_ground, 2] =255   
    Rover.worldmap[y_world_rock, x_world_rock, 2] = 0   
        

        
     # 8) Convert rover-centric pixel positions to polar coordinates
    
    
    #decide which way to go by getting the polar coordinates of the ground pixels
    ground_dist, ground_angle =to_polar_coords(x_rover_ground, y_rover_ground)
    rock_dist, rock_angles = to_polar_coords(x_rover_rock, y_rover_rock) 
    
    
             
            
     #if this conditon became true start the debugge mode        
    if debugge_mode:
        vision_image_for_display=0
       
            
        #show original image
        original_image =cv2.cvtColor(Rover.img, cv2.COLOR_BGR2RGB)
        original_image=prepare_image_for_show(original_image,'Original image') 
        
           
        #show cropped image
        Cropped_image =cv2.cvtColor(croprd_image, cv2.COLOR_BGR2RGB)
        Cropped_image=prepare_image_for_show(Cropped_image,'Cropped image') 
        
        #show wraped image
        Wrapped_image =cv2.cvtColor(wraped_image, cv2.COLOR_BGR2RGB)
        Wrapped_image=prepare_image_for_show(Wrapped_image,'Wrapped image') 
        
             
        #show gray wraped image
        Gray_Wrapped_image =cv2.cvtColor(wraped_image_gray, cv2.COLOR_BGR2RGB)
        Gray_Wrapped_image=prepare_image_for_show(Gray_Wrapped_image,'Gray wraped image') 
        
        
        #concatenate the four previous images
        first_four_images=np.concatenate((original_image,Cropped_image,Wrapped_image,Gray_Wrapped_image),axis=1)
        
        #show detected rock image
        Rover.vision_image_for_display[:,:,1]=rock_image       
        Rock_image =Rover.vision_image_for_display
        
        Rock_image=prepare_image_for_show(Rock_image,'rock image')
        
        Rover.vision_image_for_display[:,:,1]=0
        
        
        #show detected obstacle image
        Rover.vision_image_for_display[:,:,2]=obstacles_image
        
        Obstacle_image =Rover.vision_image_for_display
        
        Obstacle_image=prepare_image_for_show(Obstacle_image,'obstacle image')
        Rover.vision_image_for_display[:,:,2]=0
        
        #show detected ground image
        Rover.vision_image_for_display[:,:,0]=ground_image 
        Ground_image =Rover.vision_image_for_display
        Ground_image2=Ground_image.copy()
        
        Ground_image=prepare_image_for_show(Ground_image,'ground image')
        
        Rover.vision_image_for_display[:,:,0]=0 
            
        second_four_images=np.concatenate((Obstacle_image,Rock_image,Ground_image),axis=1)    
            
        #show the rover_coordinates and world map conversion conversion
        
        #calculation for the ground pixels
        ...
        mean_ground_dir =np.mean(angle)
        ground_arrow_length =np.mean(dist)
        
        #get the x,y coordinates of the arrow to draw it on the image
        x_arrow_ground =ground_arrow_length*np.cos(mean_ground_dir)
        y_arrow_ground =ground_arrow_length*np.sin(mean_ground_dir)
        ...
        
        #calculation for the rock pixels
        ...
        mean_rock_dir =np.mean(rock_angles)
        rock_arrow_length =np.mean(rock_dist)       
        
        #get the x,y coordinates of the arrow to draw it on the image
        x_arrow_rock =rock_arrow_length*np.cos(mean_rock_dir)
        y_arrow_rock =rock_arrow_length*np.sin(mean_rock_dir)
        ...
 
         
        if (is_rock_nav_direction) :
            #this means that we found a rock and the rover will move toward the rock
            green_color =(0,255,0)
            
            ground_image_rotated =cv2.rotate(Ground_image2,cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            if(not (math.isnan(x_arrow_rock)) and not (math.isnan(y_arrow_rock))):
                start_point =(int(ground_image_rotated.shape[1]),int(ground_image_rotated.shape[0]/2))
                end_point =(int(x_arrow_rock),int(y_arrow_rock)+int(ground_image_rotated.shape[0]/2))

                ground_image_rotated =cv2.arrowedLine(ground_image_rotated,start_point,end_point,green_color,3)

            ground_image_with_arrow=cv2.rotate(ground_image_rotated,cv2.ROTATE_180)  
            
            ground_image_with_arrow=prepare_image_for_show(ground_image_with_arrow,'ROCK Nav direction')
            
            second_four_images=np.concatenate((second_four_images,ground_image_with_arrow),axis=1)
            
        elif (is_ground_nav_direction):
            #this means that we found a rock and the rover will move toward the rock
            red_color =(0,0,255)
            #cv2.imshow('ground image',ground_image)
            
            ground_image_rotated =cv2.rotate(Ground_image2,cv2.ROTATE_90_COUNTERCLOCKWISE)
            #ground_image_rotated =cv2.cvtColor(ground_image_rotated, cv2.COLOR_BGR2RGB)
            
            if not (math.isnan(x_arrow_ground)) and not (math.isnan(y_arrow_ground)):
                start_point =(int(ground_image_rotated.shape[1]),int(ground_image_rotated.shape[0]/2))
                end_point =(int(x_arrow_ground),int(y_arrow_ground)+int(ground_image_rotated.shape[0]/2))

                ground_image_rotated =cv2.arrowedLine(ground_image_rotated,start_point,end_point,red_color,3)

            ground_image_with_arrow=cv2.rotate(ground_image_rotated,cv2.ROTATE_180) 
            
            ground_image_with_arrow=prepare_image_for_show(ground_image_with_arrow,'GROUND Nav direction')
            
            second_four_images=np.concatenate((second_four_images,ground_image_with_arrow),axis=1)
  
        #concatenate the eight images
        eight_image =np.concatenate((first_four_images,second_four_images),axis=0)
        
        cv2.imshow('DEPUGGING MODE',eight_image)
        
        cv2.waitKey(5)

 
    return Rover