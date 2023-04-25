
import numpy as np


# This is where you can build a decision tree for determining throttle, brake and steer 
# commands based on the output of the perception_step() function
def decision_step(Rover):

    # Implement conditionals to decide what to do given perception data
    # Here you're all set up with some basic functionality but you'll need to
    # improve on this decision tree to do a good job of navigating autonomously!


    # Check if we have vision data to make decisions with
    if Rover.nav_angles is not None:

        # Check for Rover.mode status
        if Rover.mode == 'forward': 

            # Check the extent of navigable terrain
            if len(Rover.nav_angles) >= Rover.stop_forward:  
                
                #every time the rover will go in the forward direction,
                # the rover_left_steer_counter will be set back to zero.
                # So, that when the rover start steering to the left, the 
                # counter will be incremanted starting from zero.  
                Rover.Rover_left_steer_counter=0

                #first we will check if there is any obstacles in the rover view area.
                #the three variables: 
                #1)Rover.obstacle_at_the_end_of_pixel:is set to true, if there is an obstacle 
                # infront the rover.
                #2)Rover.left_side_obstacle:is set to true, if there is an obstacle at the left of 
                # the rover.
                #3)Rover.right_side_obstacle: is set to true, if there is an obstacle at the right of
                # the rover.
                #so if there any obstacle in one of the directions, the rover will stop
                #by setting the throttle(accelaration) to zero ,steer (turning) to zero and setting the 
                #brake to a value (this is value was choosen without a specific reason) which equal to 10 
                #(Rover.brake_set) and changing the mode to stop
                if Rover.obstacle_at_the_end_of_pixel or Rover.left_side_obstacle or Rover.right_side_obstacle : #or Rover.two_sides_obstacles
                    Rover.throttle = 0
                    Rover.brake = Rover.brake_set
                    Rover.steer = 0
                    Rover.mode='stop'

                # else, in case there was no obstacle in the view area of the rover, then check:
                # if the rover current velocity was less than the maxmium velocity -> (this is value was
                # choosen without a specific reason, but increasing the velocity will probably require a 
                # more control over the rover driving) which equal to 2, then we will throttle the rover 
                # (accelarate the rover) by setting it to a specific value which equala 0.2 (this is value
                # was choosen without a specific reason, but increasing the throttle will probably require a 
                # more control over the rover driving).
                # else -> that means the rover is already at maximum velocity, so we are not going to accelerate
                # the rover to not exceede the maximum velocity, and ofcourse set the brake to zero to not cause  
                # the rover to stop.
                # in both cases the rover mode will still be forward. 
                else:  
                    if Rover.vel < Rover.max_vel:
                        Rover.throttle = Rover.throttle_set
                    else:
                        Rover.throttle = 0
                    Rover.brake = 0
            #   Rover.brake = 0

                #this part was done to avoid was done to avoid going in one direction in the map. In the orginal 
                #code, the rover will not go to left side of the map if it starting point was not the left. As the 
                #rover was steering in the (-15) the right direction in case of navigable terrian was less than a 
                #specified amount. This made the rover movement tend to be in the right direction.
                #to try to avoid this behaviour, a variable (Rover.count_avarage_steer)  was used to count the consecutive
                #times the rover drove in the average direction. In case the count_avarage_steer was more than 10000 (there were 
                # no specific reason for choosing this value in particular, only a large value was desired because the problem was
                # in the left part of the map which was less the the rest, so moving in the average direction for a bigger time 
                # was desired to navigate the non-left side of the map).

                #the main idea was to change the rover direction, once to the left and one to the right. This was achieved by    => try changing it only to the left
                #by storing the last_turn direction of the rover in the variable (Rover.Rover_last_turn), the direction wil be
                #alternating every time the Rover.count_avarage_steer exccede 100000.
                if (Rover.count_avarage_steer>10000) and (Rover.Rover_last_turn=='left') :#and (Rover.Rover_avarage_left_turn<30 )
                    Rover.throttle = 0
                    Rover.brake = Rover.brake_set
                    Rover.steer = 0
                    Rover.count_avarage_steer=0
                    Rover.mode="right_forward"
                elif (Rover.count_avarage_steer>10000 ) and (Rover.Rover_last_turn=='right') :#and (Rover.Rover_avarage_right_turn<30)
                    Rover.throttle = 0
                    Rover.brake = Rover.brake_set
                    Rover.steer = 0
                    Rover.count_avarage_steer=0
                    Rover.mode="left_forward"
                else:    
                    Rover.Rover_avarage_left_turn=0
                    Rover.Rover_avarage_right_turn=0
                    Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
                    Rover.count_avarage_steer+=1
                    if(Rover.steer<0):
                        Rover.Rover_last_turn='right' 
                    else:
                        Rover.Rover_last_turn='left' 
            # If there's a lack of navigable terrain pixels then go to 'stop' mode
            elif len(Rover.nav_angles) < Rover.stop_forward:
                    # Set mode to "stop" and hit the brakes!
                    Rover.throttle = 0
                    # Set brake to stored brake value
                    Rover.brake = Rover.brake_set
                    Rover.steer = 0
                    Rover.mode = 'stop'

        # If we're already in "stop" mode then make different decisions
        elif Rover.mode == 'stop':
            # If we're in stop mode but still moving keep braking
            if Rover.vel > 0.2:
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0
            # If we're not moving (vel < 0.2) then do something else
            elif Rover.vel <= 0.2:
                # Now we're stopped and we have vision data to see if there's a path forward
                if len(Rover.nav_angles) < Rover.go_forward or Rover.obstacle_at_the_end_of_pixel or Rover.left_side_obstacle or Rover.right_side_obstacle or Rover.two_sides_obstacles:
                    Rover.throttle = 0
                    # Release the brake to allow turning
                    Rover.brake = 0
                    # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
                    #Rover.steer = -15 
                    for a in range(1,int((Rover.yaw+360*2)/15)):                        
                       Rover.steer=15
                 # If we're stopped but see sufficient navigable terrain in front then go!
                if len(Rover.nav_angles) >= Rover.go_forward:
                    # Set throttle back to stored value
                    Rover.throttle = Rover.throttle_set
                    # Release the brake
                    Rover.brake = 0
                    # Set steer to mean angle
                    Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
                    Rover.mode = 'forward'

        elif Rover.mode=='left':
            if Rover.vel > 0.2:
                    Rover.throttle = 0
                    # Release the brake to allow turning
                    Rover.brake = Rover.brake_set
                    # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
                    Rover.steer = 0 # Could be
            else: # Else coast             
                    Rover.throttle =Rover.throttle_set
                    # Release the brake to allow turning
                    Rover.brake = 0
                    # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
                    Rover.steer = 15 
                    if(Rover.Rover_left_steer_counter<70):
                        Rover.Rover_left_steer_counter+=1
                    else:    
                        Rover.Rover_left_steer_counter=0
                        Rover.mode ='forward'     
                        
        elif Rover.mode=='left_forward':
            if Rover.vel > 0.2:
                    Rover.throttle = 0
                    # Release the brake to allow turning
                    Rover.brake = Rover.brake_set
                    # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
                    Rover.steer = 0 # Could be
            else: # Else coast             
                    Rover.throttle =Rover.throttle_set
                    # Release the brake to allow turning
                    Rover.brake = 0
                    # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
                    Rover.steer = 15  
                    if(Rover.Rover_avarage_left_turn <120):
                        Rover.Rover_avarage_left_turn+=1
                    else:
                        Rover.Rover_avarage_left_turn=0
                        Rover.mode ='forward'
        
        elif Rover.mode=='right_forward':
            if Rover.vel > 0.2:
                    Rover.throttle = 0
                    # Release the brake to allow turning
                    Rover.brake = Rover.brake_set
                    # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
                    Rover.steer = 0 # Could be
            else: # Else coast             
                    Rover.throttle =Rover.throttle_set
                    # Release the brake to allow turning
                    Rover.brake = 0
                    # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
                    Rover.steer = -15 
                    if(Rover.Rover_avarage_right_turn <120):
                        Rover.Rover_avarage_right_turn+=1
                    else:
                        Rover.Rover_avarage_right_turn=0
                        Rover.mode ='forward'
 
    

            if Rover.near_sample ==0:
                dist_rock = min(Rover.nav_dists)
                #check the distance if small then the rover need to rotate, else go to the rock
                if (dist_rock<10):
                    if(Rover.vel>0.2):
                        #stop the rover
                        Rover.throttle =0
                        Rover.brake = Rover.brake_set
                        Rover.steer = 0  
                    else:
                        # the rover is stopped, but the rock not in front of the rover
                        # let the rover steer toward the rock
                        Rover.throttle =0
                        Rover.brake = 0
                        Rover.steer = np.clip(np.mean(Rover.rock_angles * 180/np.pi), -15, 15) 
                        
                else:
                    #move rover to the direciton of the rock by taking the avarage of rock_angle
                    Rover.throttle =Rover.throttle_set
                    Rover.brake = 0
                    Rover.steer = np.clip(np.mean(Rover.rock_angles * 180/np.pi), -15, 15)
                    
            else:
                #then the rover is near sample, stop the rover and go to the forward mode
                Rover.throttle =0
                Rover.brake = Rover.brake_set
                Rover.steer = 0 
                
        
#*******************************************************************************************
    else:
        Rover.throttle = Rover.throttle_set
        Rover.steer = 0
        Rover.brake=0  
    # If in a state where want to pickup a rock send pickup command
    if Rover.near_sample and Rover.vel == 0 and not Rover.picking_up:
        Rover.send_pickup = True
    
    return Rover

