import json
import numpy as np
import math

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# class for rocket launch environment
class Rocket_launch_simulation:
    '''Rocket_launch_simulation is a class that simulates a rocket launch.'''

    def __init__(self, rocket_information, launch_condition):
        '''Rocket_launch_simulation(rocket_information, launch_condition) -> Rocket_launch_simulation
        Initalizes a rocket launch simulation.
        rocket_information is a JSON of the rocket performance.
        launch_condition is a dictionary of the launch conditions.'''
        # get rocket performance infromation
        self.load_rocket_information(rocket_information)

        # load launch condition
        self.launch_inclination = math.radians(launch_condition["inclination"]) # launch orbit inclination

        # enivornment constants
        self.cp = 1004.6851
        self.exp1 = ((self.cp*0.0290) / (8.3145))
        self.L = [-0.0065, 	0.0, 0.001, 0.0028, 	0.0, -0.0028, 	-0.002]
        self.Hb = [0, 11000, 20000, 32000, 470000, 51000, 71000]
        self.Tb = [288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65]
        self.MDb = [1.2250, 0.36391, 0.08803, 0.01322, 0.00143, 0.00086, 0.000064]

        self.mu = (6.6743 * (10 ** -11)) * (5.97219 * (10 ** 24))
        self.radius_earth  = 6371000
        self.earth_rotation_speed = 7.2921159 * 10**-5

        self.reset()



    def load_rocket_information(self, rocket_information):
        '''load_rocket_information(rocket_information) -> None
        sets the rocket performance based on rocket_information JSON'''
        # open the file
        rocket_file = open(rocket_information)

        self.rocket_dictionary = json.load(rocket_file) # load the JSON

        # define the rocket stage names
        self.stage_names = ["Stage " + str(stage_number + 1) for stage_number in range(self.rocket_dictionary["Number of Stages"])]

        # get the pitch rate
        self.pitch_rate = self.rocket_dictionary["Pitch Rate"]

        # Define the thrust and burn rate for each Burn
        for stage_name in self.stage_names:
            isp = self.rocket_dictionary[stage_name]["ISP"]

            # thrust and burn rate list
            self.rocket_dictionary[stage_name]["Burn Rate"] = []
            self.rocket_dictionary[stage_name]["Thrust"] = []

            # get the # of burns
            for burn_number in range(self.rocket_dictionary[stage_name]["Number of Burns"]):
                # get the thrust force and burn rate
                self.rocket_dictionary[stage_name]["Burn Rate"].append(
                    self.rocket_dictionary[stage_name]["Burns Fuel Mass"][burn_number]/self.rocket_dictionary[stage_name]["Burns Time"][burn_number]
                    )

                self.rocket_dictionary[stage_name]["Thrust"].append(isp * 9.8 * self.rocket_dictionary[stage_name]["Burn Rate"][burn_number])
            
                







    def thrust(self, time):
        '''Rocket_launch_simulation.thrust(time) -> float
        returns the thrust of the rocket depending on the time'''
        # initalize variables
        deployment_time = self.rocket_dictionary["Deployment Time"]
        
        # if payload deployed no more thruts
        if time > deployment_time:
            return 0
        
        total_thrust = 0

        # TBD ################################
        # add booster thrust
        # if haveBooster and time < boosterBurnTime:
        #     total_thrust += boosterFuelMass*9.8*boosterISP*boosterNumber/boosterBurnTime

        # Check which stage is being used at the moment
        for stage_name in self.stage_names:
            if time < self.rocket_dictionary[stage_name]["Seperation Time"]: # operating stage 
                # check which burn the rocket is in
                for burn_number in range(self.rocket_dictionary[stage_name]["Number of Burns"]):
                    if time <= self.rocket_dictionary[stage_name]["Burns Start Time"][burn_number] + self.rocket_dictionary[stage_name]["Burns Time"][burn_number] and time > self.rocket_dictionary[stage_name]["Burns Start Time"][burn_number]:
                        return total_thrust + self.rocket_dictionary[stage_name]["Thrust"][burn_number] # add the thrust from main stage and return the thrust
        return 0
                


        
    def mass(self, time, height):
        '''Rocket_launch_simulation.mass(time, height) -> float
        returns the mass of the rocket depending on the time and height'''

        # consistent thrust
        total_mass = 0 # total mass of the rocket at this time
        present_stage_name= ""

        # check if deployed the payload
        #add payload mass
        total_mass += self.rocket_dictionary["Payload Mass"]

        

        if self.rocket_dictionary["Deployment Time"] < time:
            return total_mass # return only payload mass after payload deployment time

        # check if  fairings and escape tower is still on the rocket
        if self.rocket_dictionary["Payload Fairing Seperation Height"] > height:
            total_mass += self.rocket_dictionary["Payload Fairing Mass"]
            
        if self.rocket_dictionary["Escape Tower Seperation Height"] > height:
            total_mass += self.rocket_dictionary["Escape Tower Mass"]

        # TBD ##########################################################################
        # # booster
        # if haveBooster and time < boosterBurnTime:
        #     boosterFuelRate = boosterFuelMass/boosterBurnTime
        #     totalMass += boosterNumber*(boosterGrossMass - (boosterFuelRate*time))


        # main stages
        for stage_name in self.stage_names: 
            if time < self.rocket_dictionary[stage_name]["Seperation Time"]: # stage still on the rocket
                # if the stage is before seperation and is the first stage to be so, it will be the stage which is operating
                if  present_stage_name == "": # doesn't have a defined name
                    present_stage_name = stage_name

                # add the full mass of the stage
                total_mass += self.rocket_dictionary[stage_name]["Empty Mass"] + sum(self.rocket_dictionary[stage_name]["Burns Fuel Mass"])



        # find the amount of fuel that has been burned
        for burn_number in range(self.rocket_dictionary[present_stage_name]["Number of Burns"]):
            # print(self.rocket_dictionary[stage_name]["Burns Start Time"][burn_number])
            # check if the burn is completed
            if time > self.rocket_dictionary[present_stage_name]["Burns Start Time"][burn_number] + self.rocket_dictionary[present_stage_name]["Burns Time"][burn_number]:
                # remove the fuel mass
                total_mass -= self.rocket_dictionary[present_stage_name]["Burns Fuel Mass"][burn_number]
            else:
                # check if the burn has started
                if time >= self.rocket_dictionary[present_stage_name]["Burns Start Time"][burn_number]:
                    # print(self.rocket_dictionary[stage_name]["Burn Rate"][burn_number])
                    total_mass -= (time - self.rocket_dictionary[present_stage_name]["Burns Start Time"][burn_number]) * self.rocket_dictionary[present_stage_name]["Burn Rate"][burn_number]
                


        # print(preStageNum, remainingMass)
        return total_mass


        


    def drag(self, velocity, height, g):
        '''Rocket_launch_simulation.drag(velocity, height, g) -> float
        returns the drag of the rocket in this conditon'''
        # into var
        cd0 =  self.rocket_dictionary["Coefficent of Drag"]

        # find Mach velocity (speed of sound)
        mach = 2.340*(10**-17)*(height**4) - 4.897*(10**-12)*(height**3) + 3.227*(10**-7)*(height**2) - 7.276*(10**-3)*height + 346.2

        machNumber = velocity/mach
        # get cd
        cdMain =  cd0*(300**(-((1.1-machNumber)**2))) + 0.01*machNumber + cd0
        
        # find air density
        bNum = -1
        for index in range(7):
            if (self.Hb[index] < height):
                bNum = index
                break

        if height < 0:
            bNum = 0
            
        if height > 86000:
            bNum = -1

        # print(bNum)
        if (bNum == -1):
            airDensity = 0
        else:
            airDensity = self.MDb[bNum] * math.exp((-9.80665 * 0.0289644 * (height - self.Hb[bNum])) / (8.3144598 * self.Tb[bNum]))
        
        
        # find the drag of main stage
        total_drag = (airDensity * (velocity ** 2) * cdMain * self.rocket_dictionary["Cross Section Area"]) / 2


        # TBD #########################################
        # # booster drag
        # if haveBooster:
        #     # get cd
        #     cdBooster =  cd0Booster*(300**(-((1.1-machNumber)**2))) + 0.01*machNumber + cd0Booster
        #     totalDrag += boosterNumber*(airDensity*(velocity**2)*cdBooster*boosterA)/2

        
        return total_drag



    def run_an_period(self, run_time, target_pitch_angle, gravity_turn, delta_time = 0.1):
        '''Rocket_launch_simulation.run_an_period(run_time, pitch_angle, gravity_turn, delta_time = 0.1) -> None
        runs the simulation for the given period at the given pitch_angle.'''

        for incrament in range(int(run_time/delta_time)):
            self.time += delta_time

            # use the pitch rate to pitch the rocket
            delta_pitch = target_pitch_angle - self.pitch_angle
            if delta_pitch < 0: # if pitching down
                # check if can reach target pitch angle
                if abs(delta_pitch) < (self.pitch_rate * delta_time):
                    self.pitch_angle = target_pitch_angle
                else: 
                    if gravity_turn:
                        # align flight angle and pitch angle
                        if self.flight_angle_space - self.pitch_angle < -self.pitch_rate * delta_time:

                        
                            self.pitch_angle -= self.pitch_rate * delta_time

                        elif self.flight_angle_space - self.pitch_angle > self.pitch_rate * delta_time:
                            self.pitch_angle += self.pitch_rate * delta_time
                        else:
                            self.pitch_angle = self.flight_angle_space
                    else:
                        self.pitch_angle -= self.pitch_rate * delta_time

            else: # pitching up
                # check if can reach target pitch angle
                if abs(delta_pitch) < (self.pitch_rate * delta_time):
                    self.pitch_angle = target_pitch_angle
                else: 
                    self.pitch_angle += self.pitch_rate * delta_time

            

            g = self.mu/((self.radius_earth + self.height) ** 2 ) # gravitaion acceleration

            self.thrust_force = self.thrust(self.time) # thrust

            self.present_mass = self.mass(self.time, self.height) # mass

            drag_force = self.drag(self.total_ground_speed, self.height, g)

                
            # this is the conversion angle from cartsian to polar
            converstion_angle = self.degree_position - math.pi/2


                
            # update acceleration 
            self.acceleration = np.array([
                (math.cos(self.pitch_angle + converstion_angle) * (self.thrust_force/self.present_mass)) - (math.cos(self.cartesian_flight_angle) * drag_force/self.present_mass ) + (g * math.cos(self.degree_position + math.pi)), 
                (math.sin(self.pitch_angle + converstion_angle) * (self.thrust_force/self.present_mass)) - (math.sin(self.cartesian_flight_angle) * drag_force/self.present_mass ) + (g * math.sin(self.degree_position + math.pi))
                ])

            self.total_acceleration = np.linalg.norm(self.acceleration)

            # update ground speed
            self.ground_speed += self.acceleration * delta_time
            self.total_ground_speed = np.linalg.norm(self.ground_speed)

                
            # update actual speed
            self.speed += self.acceleration * delta_time

            self.total_speed =  np.linalg.norm(self.speed)


            # update position
            self.position += self.speed * delta_time

            # update height
            self.height = np.linalg.norm(self.position) - self.radius_earth

            # update flight angle
            self.cartesian_flight_angle = np.arctan(self.speed[1]/self.speed[0])
            
            if self.speed[0] < 0:
                self.cartesian_flight_angle += math.pi
                
            self.flight_angle_space = self.cartesian_flight_angle - converstion_angle



            # update degree position
            if self.position[0] == 0:
                if self.position[1] >= 0: # above x axis
                    self.degree_position = math.pi/2
                else: # below x axis
                    self.degree_position = -math.pi/2
            else:
                self.degree_position = np.arctan(self.position[1]/self.position[0])
                if self.position[0] < 0:
                    self.degree_position += math.pi

            self.degree_position %= 2 * np.pi

            # find polar speed and acceleration
            acceleration_direction =  np.arctan(self.acceleration[1]/self.acceleration[0])
            
            if self.acceleration[0] < 0:
                acceleration_direction += math.pi
                
            acceleration_direction -= converstion_angle


            self.polar_acceleration = np.array([math.cos(acceleration_direction),  math.sin(acceleration_direction)]) * self.total_acceleration
            self.polar_speed = np.array([math.cos(self.flight_angle_space),  math.sin(self.flight_angle_space)]) * self.total_speed

            # get the distance from the launch site in terms of the surface of earth
            self.downrange_distance = (self.starting_degree_postion - self.degree_position) * self.radius_earth

            self.downrange_list.append(self.downrange_distance / 1000)

            # save new recording value
            self.time_list.append(self.time)

            self.height_list.append(self.height/1000)
           
            self.total_acceleration_list.append(self.total_acceleration)
            self.total_speed_list.append(self.total_speed)
            self.speed_list.append(self.speed)

            # self.position_list.append(self.position) 
            # print(self.position_list.shape)
            self.position_x_list.append(self.position[0]) 
            self.position_y_list.append(self.position[1]) 


            
            self.mass_list.append(self.present_mass)
            self.thrust_list.append(self.thrust_force)

            self.pitch_angle_list.append(math.degrees(self.pitch_angle))

            self.flight_angle_space_list.append((math.degrees(self.flight_angle_space)+ 90) % 360 - 90)
            

            # check if hit the ground
            if self.height < 0:
                self.hit_ground = True
                break



    def get_states(self):
        '''Rocekt_launch_simulation.get_states() -> list
        returns an array of the rockets state'''
        return [self.time, self.thrust_force, self.present_mass, self.acceleration[0], self.acceleration[1], self.speed[0], self.speed[1], self.flight_angle_space, self.pitch_angle, self.degree_position, self.height, self.downrange_distance, self.position[0], self.position[1]]
    

    def step(self, target_pitch_angle, gravity_turn):
        '''Rocekt_launch_simulation.step() -> list
        returns the new state given a step'''
        self.run_an_period(0.1, target_pitch_angle, gravity_turn)
        return self.get_states()


    def find_orbital_perameter(self):
        '''Rocekt_launch_simulation.find_orbital_perameter() -> list
        returns the orbits perigee, apogee, and eccentricity.'''

        # find important values
        distance = self.height + self.radius_earth
        h = math.cos(self.flight_angle_space) * self.total_speed*distance


        seni_major_axis = self.mu * distance/((2 * self.mu)-(distance * ((self.total_speed) ** 2)))

        eccentricity = math.sqrt(1 - (h ** 2 * (2 * self.mu - distance * (self.total_speed) ** 2) / (self.mu ** 2 * distance)))

        # calculate perigee and apogee
        perigee = seni_major_axis * (1 - eccentricity) - self.radius_earth
        apogee = seni_major_axis * (1 + eccentricity) - self.radius_earth

        return [perigee, apogee, eccentricity]
    

    def display_data(self):
        '''Rocekt_launch_simulation.display_data() -> list
        display the data.'''
        figure, axis = plt.subplots(2, 5) 

        axis[0,0].plot(self.time_list, self.height_list)
        # axis[0,0].plot(tList, hTestList)
        axis[0,0].set_xlabel('t (s)')
        axis[0,0].set_ylabel('h (km)')
        axis[0,0].legend(['x', 'y'], shadow=True)
        axis[0,0].set_title('Rocket Height')

        axis[0,1].plot(self.time_list, self.total_speed_list)
        axis[0,1].set_xlabel('t (s)')
        axis[0,1].set_ylabel('v (m/s)')
        axis[0,1].legend(['x', 'y'], shadow=True)
        axis[0,1].set_title('Rocket Velocity')

        axis[1,0].plot(self.time_list, self.total_acceleration_list)
        axis[1,0].set_xlabel('t (s)')
        axis[1,0].set_ylabel('a (m/s^2)')
        axis[1,0].legend(['x', 'y'], shadow=True)
        axis[1,0].set_title('Rocket Acceleration')

        axis[1,1].plot(self.time_list, self.thrust_list)
        axis[1,1].set_xlabel('t (s)')
        axis[1,1].set_ylabel('Force (N)')
        axis[1,1].legend(['x', 'y'], shadow=True)
        axis[1,1].set_title('Thrust')

        axis[0,2].plot(self.time_list, self.mass_list)
        axis[0,2].set_xlabel('t (s)')
        axis[0,2].set_ylabel('Mass (kg)')
        axis[0,2].legend(['x', 'y'], shadow=True)
        axis[0,2].set_title('Rocket Mass')

        axis[1,2].plot(self.time_list, self.flight_angle_space_list)
        axis[1,2].set_xlabel('t (s)')
        axis[1,2].set_ylabel('Angle (Deg)')
        axis[1,2].legend(['x', 'y'], shadow=True)
        axis[1,2].set_title('Flight Angle')

        axis[0,3].plot(self.time_list, self.pitch_angle_list)
        axis[0,3].set_xlabel('t (s)')
        axis[0,3].set_ylabel('Angle (Deg)')
        axis[0,3].legend(['x', 'y'], shadow=True)
        axis[0,3].set_title('Pitch Angle')

        #earth

        center = (0, 0)
        radius = self.radius_earth
        circle = Circle(center, radius, color='black', fill=False)

        axis[1,3].set_aspect('equal')
        axis[1,3].add_patch(circle)
        axis[1,3].plot(self.position_x_list, self.position_y_list)
        axis[1,3].set_title('Orbit')


        axis[0,4].plot(self.downrange_list, self.height_list)
        axis[0,4].set_xlabel("Downrange (km)")
        axis[0,4].set_ylabel("Height (km)")
        axis[0,4].set_title('Trajectory')




        plt.show()

    def reset(self):
         # initalize simulation variables
        self.time = 0
        self.hit_ground = False

        self.starting_degree_postion = math.pi/2

        self.degree_position = self.starting_degree_postion # this is the angle in the 2D plane of the orbit
        
        # height
        self.height = 0

        # acceleration
        self.acceleration = np.array([0.0, -self.mu/((self.radius_earth + self.height) ** 2 )])

        self.polar_acceleration = np.array([0.0, -self.mu/((self.radius_earth + self.height) ** 2 )])

        self.total_acceleration = 0

        # initalize the inital speed of the rocket. Factor in earth rotation
        self.speed = np.array([
            (self.earth_rotation_speed * math.cos(self.launch_inclination) * (self.radius_earth + self.height))  * math.cos(self.degree_position - math.pi/2),  
            (self.earth_rotation_speed * math.cos(self.launch_inclination) * (self.radius_earth + self.height))  * math.sin(self.degree_position - math.pi/2)
            ])
        
        self.polar_speed = np.array([
            (self.earth_rotation_speed * math.cos(self.launch_inclination) * (self.radius_earth + self.height))  * math.cos(self.degree_position - math.pi/2),  
            (self.earth_rotation_speed * math.cos(self.launch_inclination) * (self.radius_earth + self.height))  * math.sin(self.degree_position - math.pi/2)
            ])

        # ground speed
        self.ground_speed = np.array([0.0, 0.0])

        self.total_ground_speed = 0
        self.total_speed = np.sum(self.speed)

        self.flight_angle_earth = math.pi/2
        self.flight_angle_space = math.pi/2
        self.cartesian_flight_angle = 0.0
        self.pitch_angle =  math.pi/2
        self.height = 0.0
        self.downrange_distance = 0.0 
        self.downrange_list = []

        # rocket values
        
        self.thrust_force = 0.0 # thrust

        self.present_mass = self.mass(0, 0) # mass

        # position
        self.position = np.array([math.cos(self.degree_position) *self.radius_earth, math.sin(self.degree_position) *self.radius_earth])

        # value recording list
        self.position_x_list= []
        self.position_y_list= []
        self.height_list = []
        self.total_acceleration_list = []
        self.speed_list = []
        self.total_speed_list = []
        self.time_list = []
        self.flight_angle_space_list = []
        self.pitch_angle_list = []
        self.degree_position_list = []
        self.thrust_list = []
        self.mass_list = []

        return self.get_states()


import os

file_name = "Starship2.json"

test = Rocket_launch_simulation(file_name, {"inclination": 0.453786})

# open a file to save trajectory
output_file = open("30_increment_file2_cart.csv", "w")

# pitch_target_angle = {20: math.radians(80), 70: math.radians(50), 130: math.radians(30), 310: math.radians(10), 400: math.radians(0), 410:  math.radians(0)}
# pitch_target_angle = {20: math.radians(80), 52: math.radians(60), 100: math.radians(40), 210: math.radians(20), 300: math.radians(10), 410:  math.radians(0)}
# pitch_target_angle = {20: math.radians(80), 52: math.radians(60), 120: math.radians(30), 300: math.radians(10), 410:  math.radians(0)}
# pitch_target_angle = {10: math.radians(80), 32: math.radians(60), 120: math.radians(30), 300: math.radians(10), 400:  math.radians(0)}
# pitch_target_angle = {20: math.radians(60), 130: math.radians(30), 325:  math.radians(0)}
pitch_target_angle = {20: math.radians(60), 150: math.radians(30), 320:  math.radians(0)}
# pitch_target_angle = {30: math.radians(60), 110: math.radians(30), 320:  math.radians(0)}
# pitch_target_angle = {10: math.radians(60), 150: math.radians(30), 300:  math.radians(0)}
# pitch_target_angle = {25: math.radians(60), 120: math.radians(30), 315:  math.radians(0)}
# pitch_target_angle = {22: math.radians(-80), 100: math.radians(0)}

state_name = ["time", "thrust", "mass", "acceleration x", "acceleration y", "speed x", "speed y", "flight angle", "pitch angle", "degree position", "height", "downrange distance", "position x", "position y"]

output_file.writelines("present " + ",present ".join(state_name) + ",pitch angle,next " + ",next ".join(state_name) + "\n")

pitch_angle = math.radians(90)
gravity_turn = False
# for time in range(0, 7):
for time in range(0, 4561):
# for time in range(0, 100000):
    time /= 10
    if round(time, 2) % 1 == 0 and pitch_target_angle.get(int(time)) != None:
        pitch_angle = pitch_target_angle.get(int(time)) 
        # gravity_turn = True
    present_state = ", ".join(str(x) for x in test.get_states())
    
    next_state = ", ".join(str(x) for x in test.step(pitch_angle, gravity_turn))

    output_file.writelines(present_state + ", " + str(pitch_angle) + ", " + next_state + "\n")

output_file.close()
print(test.find_orbital_perameter())
test.display_data()
