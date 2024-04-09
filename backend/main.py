import asyncio
import json
import websockets
import numpy as np 

class Robot:
    def __init__(self):
        with open('config.json', 'r') as config_file:
            self.state = json.load(config_file)

        self.pe = np.array([0, 0, 0])
        self.delta = 0.01
        np.set_printoptions(precision=None, suppress=False)

        self.state_lock = asyncio.Lock()
        self.moving_base = False


    def mm_to_meters(self, value):
        return value/1000
    
    def meters_to_mm(self,value):
        return value*1000 
    
    def rad_to_deg(self,value):
        return (value * 180/np.pi) 

    def deg_to_rad(self,value):
        return (value * np.pi/180)
        
    # Set functions
    async def set_the_base_position(self, new_position):
        async with self.state_lock:
            new_position = [ self.meters_to_mm(x) for x in new_position]
            self.state["position"] = new_position

    async def set_the_xz_base_postion(self, x,y):
        async with self.state_lock:
            self.state["position"][0] =  self.meters_to_mm(x)
            self.state["position"][2] =  self.meters_to_mm(y)

    async def set_dynamic_state_manipulator(self, state_update):
        # Use the update method such that non valid keys do not result in an error
        async with self.state_lock:
            for key, value in state_update["data"].items():
                if key in self.state["dynamicState"]:

                    if key in ["lift", "gripper"]:
                        value =  self.meters_to_mm(value)
                    else:
                        value = self.rad_to_deg(value)

                    self.state["dynamicState"][key] = value
                else:
                    print(f"Key {key} not found in dynamic state")


    def convert_dict_of_mm_list_to_meter(self, value_dict):
        intermediate_dict = {}
        for key, value in value_dict.items():
            intermediate_dict[key] = [self.mm_to_meters(x) for x in value]

        return intermediate_dict

    def _set_state_directly(self, state_update):
        self.state.update(state_update)
    
    # Get functions
    async def get_dynamic_actuation_state(self, destructure = False):
        # Get the angles from the dynamic state 
        async with self.state_lock:
            base_angle =  self.deg_to_rad(self.state.get('dynamicState', {}).get('base', 0.0))
            elbow_angle = self.deg_to_rad(self.state.get('dynamicState', {}).get('elbow', 0.0))
            wrist_angle = self.deg_to_rad(self.state.get('dynamicState', {}).get('wrist', 0.0))
            lift_rise =   self.mm_to_meters(self.state.get('dynamicState', {}).get('lift', 0.0))
            gripper =     self.mm_to_meters(self.state.get('dynamicState', {}).get('gripper', 0.0))

        if destructure:
            return base_angle, elbow_angle, wrist_angle, lift_rise, gripper
        else : 
            return np.array([base_angle, elbow_angle, wrist_angle, lift_rise, gripper])
        
    async def get_dimensions(self): 
        async with self.state_lock:
            dimensions = self.state.get('dimensions', {})
            dimensions = self.convert_dict_of_mm_list_to_meter(dimensions)
        return dimensions
    
    async def get_state_for_ws(self):
        endpoint = await self.end_effectors_position()
        position = await self.get_base_position()
        dimension = await self.get_dimensions()
        dynamicState = await self.get_dynamic_actuation_state() 
        orientation = await self.get_orientation()
        state =   {
        "base": dynamicState[0],
        "elbow": dynamicState[1],
        "wrist": dynamicState[2],
        "lift": dynamicState[3],
        "gripper": dynamicState[4]
        }
        message = {
            "action": "get_state",
            "data": {
                "dimensions": dimension,
                "dynamicState": state,
                "endpoint" : endpoint.tolist(),
                "position": position,
                "orientation" : orientation 
            }
        }
        return message
    
    async def set_orientation(self, orientation):
        async with self.state_lock:
            self.state["orientation"] = self.rad_to_deg(orientation)

    async def get_orientation(self):
        async with self.state_lock:
            orientation = self.state.get('orientation', 0) 
        return self.deg_to_rad(orientation)
    
    async def get_base_position(self):
        async with self.state_lock:
            position = self.state.get('position', [0,0,0]) 
        return [self.mm_to_meters(x) for x in position]
    
    async def get_numpy_base_xz_position(self):
        position = await self.get_base_position()
        return np.array([position[0], position[2]])

    def get_gains(self):
        Kp_angles = 1 
        Kp_linear = 2

        column = np.ones((7,3))
        column[:5,0] *= Kp_angles
        column[5:,0] *= Kp_linear 

        column[:5,1] *= Kp_angles * 0.4
        column[5:,1] *= Kp_linear * 0.4

        column[:5,2] *= Kp_angles * 1.1
        column[5:,2] *= Kp_linear * 1.1

        return column
    
        
    async def get_controlable_state(self):
        dynamic_actuation_state = await self.get_dynamic_actuation_state(destructure = False)
        base_position = await self.get_numpy_base_xz_position()
        return np.concatenate((dynamic_actuation_state, base_position), axis=0)
    
    async def get_dimensions_of_the_links(self):
        dimension = await self.get_dimensions()
        base_dim, lift_dim = dimension['base'], dimension['lift']
        upper_arm_dim, lower_arm_dim = dimension['upperArm'], dimension['lowerArm']
        palm_dim , left_dim = dimension['palm'], dimension['leftFinger']
        return base_dim, lift_dim, upper_arm_dim, lower_arm_dim, palm_dim, left_dim
    
    # Helper functions
    async def calc_setpoint(self, data):  
        setpoint = np.array(data["setpoint"])
        orientation = await self.get_orientation()
        base_position = await self.get_base_position()

        # Transform the setpoint to the Robot frame
        base_position = np.array(base_position)
        setpoint[:3] = setpoint[:3] - base_position
        rot_matrix = np.array([[np.cos(orientation), 0, -np.sin(orientation)], [0, 1, 0], [np.sin(orientation), 0, np.cos(orientation)]])
        setpoint[:3] = rot_matrix @ setpoint[:3]
        setpoint[2] *= -1
        if len(setpoint) == 4:
            setpoint[3] -= orientation

        
        _, _, upper_arm_dim, lower_arm_dim, palm_dim, _ = await self.get_dimensions_of_the_links()

        # Account for the dimensions of the links
        setpoint[1] += (upper_arm_dim[1]/2 + lower_arm_dim[1] + palm_dim[1]/2)

        return setpoint
    
    async def calc_target_state(self, data):
        # setpoint_actuation = [base_angle, elbow_anblge, wrist_anlge, lift_pos, gripper_pos, base_x, base_y]
        # setpoint = [endx, endy, endz, phi, basex, basez]
       
        if "setpoint_actuation" in data:
            return np.array(data["setpoint_actuation"])
        else:
            setpoint = await self.calc_setpoint(data)
            correct_angles = await self.calc_inverse_kinematics(setpoint) # base , elbow, wrist
           
            if not isinstance(correct_angles, np.ndarray):
                return -1
            
            dynamic_actuation_state = await self.get_dynamic_actuation_state(destructure=True)
            current_gripper_pos = dynamic_actuation_state[4]
            correct_actuation_position = np.array([setpoint[1], current_gripper_pos]) #lift, gripper
            current_base_pos = await self.get_numpy_base_xz_position()  # base_x, base_y

        target = np.concatenate((correct_angles, \
                                 correct_actuation_position,
                                 current_base_pos), axis=0)
        return target
    
    async def calc_forward_kinematics(self):
        # Get the link dimensions 
        uppper_link_length, lower_link_length, palm_link_length = await self.calc_link_lengths()

        # Get the angles 
        base_angle, elbow_angle, wrist_angle,lift_height,_ = await self.get_dynamic_actuation_state(destructure = True)

        # Get the dimensions of the links
        base_dim, _, upper_arm_dim, lower_arm_dim, palm_dim, _ = await self.get_dimensions_of_the_links()

        end_position = np.array([0.0,0.0,0.0])

        end_position[0] = uppper_link_length * np.cos(base_angle) + lower_link_length * np.cos(base_angle + elbow_angle) + palm_link_length * np.cos(base_angle + elbow_angle + wrist_angle)
        end_position[1] = lift_height + base_dim[1] - upper_arm_dim[1] - lower_arm_dim[1] - palm_dim[1]
        end_position[2] = uppper_link_length * np.sin(base_angle) + lower_link_length * np.sin(base_angle + elbow_angle) + palm_link_length * np.sin(base_angle + elbow_angle + wrist_angle)
        return end_position

    async def end_effectors_position(self):
        end_position = await self.calc_forward_kinematics()
        orientation = await self.get_orientation()

        # Transform the end position to the base frame
        position = await self.get_base_position()
        rot_matrix = np.array([[np.cos(orientation), 0, -np.sin(orientation)], [0, 1, 0], [np.sin(orientation), 0, np.cos(orientation)]])
        inv_rot_matrix = np.linalg.inv(rot_matrix)
        end_position = inv_rot_matrix @ end_position
        end_position += np.array(position)

        return end_position
    
    async def calc_inverse_kinematics(self, end_point):
        # Links in meters

        angles = np.zeros(3)
        phi = 0 if len(end_point) < 4 else end_point[3]
        a1, a2, a3 = await self.calc_link_lengths()

        p2x = end_point[0] - a3 * np.cos(phi)
        p2z = end_point[2] - a3 * np.sin(phi)
        dist = np.sqrt(p2x**2 + p2z**2)
        
        # Cosine rule
        cos2 = (dist**2 - a1**2 - a2**2)/(2 * a1 * a2)
        
        if cos2 > 1 or cos2 < -1:
            print("Unreachable")
            return -1
        
        # Radius
        sin2 = np.sqrt(1 - cos2**2)
       
        angles[1] = np.arctan2(sin2, cos2)
        angles[0] = np.arctan2(p2z, p2x) - \
                    np.arctan2(a2 * np.sin(angles[1]), a1 + a2 * np.cos(angles[1]))
        angles[2] = phi - angles[0] - angles[1] 

        return angles

    async def calc_link_lengths(self):
        _, lift, upper_arm, lower_arm, palm,_ = await self.get_dimensions_of_the_links()
        uppper_link_length = upper_arm[0] - lower_arm[1]/2 + lift[0]/2
        lower_link_length = lower_arm[0] - palm[1]/2 - upper_arm[1]/2
        palm_link_length = palm[0] - lower_arm[1]/2


        return uppper_link_length, lower_link_length, palm_link_length  
    
    def clip_speed(self, speed, vel_prev):
        
        max_speed_joints = 5
        max_speed_linear = 5

        # clip for the accelarion
        if isinstance(vel_prev,np.ndarray):
            acceleration_max = 10 #m.s^-2
            acc = (speed - vel_prev)/self.delta
            acc[:3] = np.clip(acc[:3], -acceleration_max, acceleration_max)
            acc[3:] = np.clip(acc[3:], -acceleration_max * 3, acceleration_max * 3)
            speed = vel_prev + acc * self.delta

        # clip max speeds
        speed[:3] = np.clip(speed[:3], -max_speed_joints, max_speed_joints)
        speed[3:] = np.clip(speed[3:], -max_speed_linear, max_speed_linear)


        return speed
    
    # Print functions 
    def print_state(self):
        print(self.state)

    def print_forward_kinematics(self):
        print(self.calc_forward_kinematics())

    async def return_to_origin(self):
        setpoint_actuation = await self.get_controlable_state()
        setpoint_actuation[-2:] = np.array([0,0]) 
        data = { "setpoint_actuation": setpoint_actuation}
        await self.control_pid(data)

    # Control functions 
    async def dance(self, home = False):
        self.moving_base = True
        time = 25
        
        
        # Create a path for the base
        length = int(time/self.delta)
        straight = np.linspace(0, 4, int(length/4))
        frequency = 0.2

        for i in straight:
            await asyncio.sleep(self.delta)
            new_position = [0, 0, i]
            await self.set_the_base_position(new_position)


        rot = await self.get_orientation()
        for i in range(length):
            await asyncio.sleep(self.delta)
            
            x = 4 * np.sin(frequency * i * self.delta) 
            y = abs(2* np.sin(i * self.delta))
            z = 4 * np.cos(frequency * i * self.delta)
            rot += 0.001
            

            new_position = [x, y, z]
            await self.set_the_base_position(new_position)
            await self.set_orientation(rot)
        
        self.moving_base = False

    async def clip_actuation(self, actuation):
        # Actuation is [base, elbow, wrist, lift, gripperm base_x, base_y]

        # Dimensions
        _, lift_dim, upper_arm_dim, lower_arm_dim, palm_dim, left_dim = await self.get_dimensions_of_the_links()
        
        # min lift
        min_lift =  left_dim[1] + palm_dim[1] + lower_arm_dim[1] + upper_arm_dim[1]/2
        max_lift =  lift_dim[1] - upper_arm_dim[1] 

        # min gripper
        min_gripper = left_dim[0]
        max_gripper = palm_dim[0] - min_gripper
        
        actuation[3] = np.clip(actuation[3], min_lift, max_lift)
        actuation[4] = np.clip(actuation[4], min_gripper, max_gripper)
        return actuation
    

    async def control_pid(self, data):
        # Placeholder valuse
        pe_prev = np.zeros(7)
        pe_int = np.zeros(7)
        self.pe = np.ones(7)
        vel_prev = None 
        while np.linalg.norm(self.pe) > 0.01 or self.moving_base:

            # Get target and current state
            target_state = await self.calc_target_state(data)

            if isinstance(target_state, int):
                continue
                

            # Make sure the target state is reachable
            target_state = await self.clip_actuation(target_state)
            current_state = await self.get_controlable_state()
            
            # Get the actuations
            self.pe = target_state - current_state

            # normalize the angles
            self.pe[:3] = np.arctan2(np.sin(self.pe[:3]), np.cos(self.pe[:3]))

            # Get the gains
            gains = self.get_gains()
            # Calculate the velocity
            pe_diff = self.pe - pe_prev

            vel = gains[:,0] * self.pe + gains[:,1] * pe_diff + gains[:,2] * pe_int
            vel = self.clip_speed(vel, vel_prev)

            # Update the state
            new_state = current_state + vel * self.delta

            await self.set_dynamic_state_manipulator({ "data" : {
                "base": new_state[0],
                "elbow":  new_state[1],
                "wrist": new_state[2],
                "lift": new_state[3],
                "gripper": new_state[4]
            }}
            )
            if not self.moving_base:
                await self.set_the_xz_base_postion(new_state[5], new_state[6])

            print(self.pe)

            pe_prev = self.pe
            pe_int += pe_diff
            vel_prev = vel
            await asyncio.sleep(self.delta/20)


        return




# The robot instance should be accessible in the coroutine functions
robot = Robot()

async def message_handler(websocket, path):
    try:
        
        async for message in websocket:
            # Your message handling logic
            message_data = json.loads(message)
            
            if message_data['action'] == "control_state":
                # Ensure that message_data is correctly parsed and used here
                robot._set_state_directly(message_data)  # Assuming control_state expects a dict
            
            elif message_data['action'] == "get_dimensions":
                await websocket.send(json.dumps(await robot.get_dimensions()))

            elif message_data['action'] == "get_initial_state":
                await websocket.send(json.dumps(await robot.get_state_for_ws()))
            
            elif message_data['action'] == "update_dynamic_state":
                await robot.set_dynamic_state_manipulator(message_data)
            
            elif message_data['action'] == "forward_kinematics":
                await websocket.send(json.dumps(await robot.calc_forward_kinematics().tolist()))
            
            elif message_data['action'] == "print_end_effector_position":
                await robot.print_forward_kinematics()
            
            elif message_data['action'] == "print_state":
                robot.print_state()

            elif message_data['action'] == "control_pid":
                await robot.control_pid(message_data['data'])
            
            elif message_data['action'] == "return_to_origin":
                await robot.return_to_origin()

            elif message_data['action'] == "move_base":
                await robot.dance()
            
            elif message_data['action'] == "move_base_and_control_pid":
                await asyncio.gather(
                robot.control_pid(message_data['data']),
                robot.dance()
            )
                
            else:
                print(f"Unknown action: {message_data['action']}: {message_data}")
                print(f"end_affector_position: {await robot.calc_forward_kinematics()}")


    except websockets.exceptions.ConnectionClosedError as e:
        print(f"Connection closed with error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # Perform any necessary cleanup here
        print("Connection handler terminated")

async def send_crane_state(websocket, path):
    while True:
        await websocket.send(json.dumps(await robot.get_state_for_ws()))
        await asyncio.sleep(0.02 )  # 20ms

async def handler(websocket, path):
    # Make sure the functions are wrappend and return a Future just like a Promise in JS
    consumer_task = asyncio.ensure_future(message_handler(websocket, path))
    producer_task = asyncio.ensure_future(send_crane_state(websocket, path))
    
    # Run the two tasks concurrently
    done, pending = await asyncio.wait(
        [consumer_task, producer_task],
        return_when=asyncio.FIRST_COMPLETED,
    )

    # Because we don't return when asyncio returns something has gone wrong
    # and we need to cancel all the other tasks
    for task in pending:
        task.cancel()

# Initialize the WebSocket server
async def main():
    start_server = websockets.serve(handler, "localhost", 8765)
    await start_server
    await asyncio.Future()  # Run forever

# Run the server
if __name__ == "__main__":
    asyncio.run(main())
