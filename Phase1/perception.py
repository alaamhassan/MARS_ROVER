import numpy as np
import cv2

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160)):
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
 
    
#*********************************************************    
#begin editing 
#function to calculate the source and destination point for the image wraping

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
    # Perform perception steps to update Rover()
    # TODO: 
    # NOTE: camera image is coming to you in Rover.img
    
    #see obstacles location for driving the rover in the correct direction
    #detect obstacles in the image
    #numpy_Rover_image =np.array(Rover.img)
    #obstacles_in_the_original_image=detect_obstecals(Rover.img) 
    
              
    # 1) Define source and destination points for perspective transform 
    
    dst=5
    bottom_offset=5
    croprd_image=Rover.img[90:150,:]
    #plt.imshow(croprd_image)
    source=np.float32([[14,140],
                      [300,140],
                      [200,95],
                      [120,95]])
    
    destination =np.float32([[Rover.img.shape[1]/2-dst,Rover.img.shape[0]-bottom_offset],
                             [Rover.img.shape[1]/2+dst,Rover.img.shape[0]-bottom_offset],
                             [Rover.img.shape[1]/2+dst,Rover.img.shape[0]-2*dst-bottom_offset],
                             [Rover.img.shape[1]/2-dst,Rover.img.shape[0]-2*dst-bottom_offset]])
    
    #source,destination =calc_source_and_destination(Rover.img)
    # 2) Apply perspective transform
    wraped_image=perspect_transform(Rover.img,source,destination)
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    
    wraped_image_gray=wraped_image.copy()
    wraped_image_gray[wraped_image_gray>0]=255
    wraped_image_gray[wraped_image_gray==0]=0
    wraped_image_gray =cv2.cvtColor(wraped_image_gray,cv2.COLOR_RGB2GRAY)
    
    #identify obstecals
    obstacles_image = cv2.bitwise_and(detect_obstecals(wraped_image),wraped_image_gray,mask=None)
    #identiy rocks
    rock_image =cv2.bitwise_and(detect_rock(wraped_image),wraped_image_gray,mask=None)
    #identify ground 
    ground_image =cv2.bitwise_and(color_thresh(wraped_image),wraped_image_gray,mask=None)

    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
        # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
        
    #put the obstacles in the r channel
    Rover.vision_image[:,:,0]=obstacles_image #(obstacles_image*255-ground_image*255)
    
    #put the rock in the g channel
    Rover.vision_image[:,:,1]=rock_image #*255
    
    #put the ground in the b channel
    Rover.vision_image[:,:,2]=ground_image #*255
     
    # 5) Convert map image pixel values to rover-centric coords
    x_rover_obstacles, y_rover_obstacles =rover_coords(obstacles_image)
    x_rover_rock, y_rover_rock =rover_coords(rock_image)
    x_rover_ground, y_rover_ground =rover_coords(ground_image)
        
    # 6) Convert rover-centric pixel values to world coordinates
    xpos =Rover.pos[0] # current Rover x position
    ypos =Rover.pos[1] # current Rover y position
    yaw =Rover.yaw
    world_size =Rover.worldmap.shape[0] #shape[0] to get the length of the worldmap
    scale =10 #3 is the dist
    x_world_obstacles, y_world_obstecals = pix_to_world(x_rover_obstacles, y_rover_obstacles, xpos, ypos, yaw, world_size, scale)
    x_world_rock, y_world_rock =pix_to_world(x_rover_rock, y_rover_rock, xpos, ypos, yaw, world_size, scale)
    x_world_ground, y_world_ground =pix_to_world(x_rover_ground, y_rover_ground,xpos, ypos, yaw, world_size, scale)
        
    # 7) Update Rover worldmap (to be displayed on right side of screen)
    Rover.worldmap[y_world_obstecals, x_world_obstacles, 0] = 255  
    Rover.worldmap[y_world_rock, x_world_rock, 0] = 0
    Rover.worldmap[y_world_rock, x_world_rock, 1] = 255  
    Rover.worldmap[y_world_ground, x_world_ground, 1]=0
    Rover.worldmap[y_world_ground, x_world_ground, 2] =255   
    Rover.worldmap[y_world_obstecals, x_world_obstacles, 2] =0
  
    # 8) Convert rover-centric pixel positions to polar coordinates
    
    #decide which way to go by getting the polar coordinates of the ground pixels
    dist, angle =to_polar_coords(x_rover_ground, y_rover_ground)

    #update the Rover.nav_angle and nav_dist
    Rover.nav_angles =angle
    Rover.nav_dists =dist
        
       
 
    return Rover
