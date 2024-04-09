

import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import Box from '@mui/material/Box';
import TextField from '@mui/material/TextField';
import IconButton from '@mui/material/IconButton';
import CameraIcon from '@mui/icons-material/CameraEnhance';
import Button from '@mui/material/Button'; // Import Button
import styles from '../styles/Home.module.css';
import favicon from '../public/monumental_favicon.png';
import { left } from '@popperjs/core';
import { Typography } from '@mui/material';
import { send } from 'process';


const Home: React.FC = () => {
  const [inputValue, setInputValue] = useState<string>('');
  const canvasRef = useRef<HTMLDivElement>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const [cameraPosition, setCameraPosition] = useState({ x: 15, y: 28, z: 20 });
  const [ws, setWs] = useState<WebSocket | undefined>(undefined); // Using a ref did nor work for the websocket
  const [messages, setMessages] = useState('');
  const sceneRef = useRef<THREE.Scene | null>(null);
  const [craneState, setCraneState] = useState(null);
  const [setpoint, setSetpoint] = useState({x : 7.1, y: 0, z: 7.1, phi : 0});
  const [endpoint, setEndpoint] = useState({x : 0, y: 3.5, z: 0});
  const [actuation, setActuation] = useState({base: 0, elbow: 0, wrist: 0, lift: 0, gripper: 0});
     
    // Change the camera postiion when the cameraPosition state changes
    useEffect(() => {
      if (cameraRef.current) {
        const { x, y, z } = cameraPosition;
        cameraRef.current.position.set(x, y, z);
      }
    }, [cameraPosition]);


    // Create websocket connection
    useEffect(() => {
      const ws = new WebSocket('ws://localhost:8765');
      
      setWs(ws);

      ws.onopen = () => {
        console.log('Connected to the server');
        
        ws.send(
          JSON.stringify({
            "action": "get_initial_state"
          })
        );
      };

      ws.onmessage = (event) => {
        handleMessages(event.data);
        setMessages(event.data);
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
      };

      ws.onclose = () => {
        console.log('WebSocket connection closed');
      };

      return () => {
        ws.close();
      };
    }, []);

    useEffect(() => {

      if (sceneRef.current && craneState) {
      const scene = sceneRef.current;
      
      const staticComponents = scene.children.filter(child => child.type === 'DirectionalLight' || child.type === 'AmbientLight' || child.type === 'GridHelper' || child.type === 'AxesHelper' );

      // Clear the scene, but preserve static components like lights
      while(scene.children.length > 0){ 
        scene.remove(scene.children[0]); 
      }

      // Re-add static components back to the scene
      staticComponents.forEach(component => scene.add(component));

      
      const palm = renderCrane(sceneRef.current); // Use sceneRef.current to access the scene
    
      const x = craneState.endpoint[0];
      const y = craneState.endpoint[1];
      const z = craneState.endpoint[2];
      cameraRef.current?.lookAt(x, y, z);

      setEndpoint({ x: x, y: y, z: z })
    }
  }, [craneState]);


  useEffect(() => {
    if (!canvasRef.current) return;

    // Create the scene
    const { scene, camera, renderer } = initThreeScene();
    cameraRef.current = camera;

    // create the crane
    const palm = renderCrane(scene);


    animate(camera, scene, renderer);
    window.addEventListener('resize', () => onResize(camera, renderer));
    
    return () => {
      window.removeEventListener('resize', () => onResize(camera, renderer));
      canvasRef.current?.removeChild(renderer.domElement);
    };
  }, []);
    
  const metersToMm = (meters: number) => meters * 1000;
  const mmToMeters = (mm: number) => mm / 1000;
  const degToRad = (deg: number) => deg * Math.PI / 180;
  const radToDeg = (rad: number) => rad * 180 / Math.PI;
    
  // Update the setpoint state when the input fields change
  const handleSetpointChange = (axis: 'x' | 'y' | 'z' | 'phi', value: string) => {
    setSetpoint((prevSetpoint) => ({
      ...prevSetpoint,
      [axis]: parseFloat(value),
    }));
  };

  const sendMessage = (json : object) => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      const message = JSON.stringify(json);

      ws.send(message);
    } else {
      console.log("WebSocket is not open.");
    
    }
  }

  const handleMessages = (message: string) => {
    const data = JSON.parse(message);
    if (data["action"] === "get_state"){
      setCraneState(data["data"]);
    }
  }

  const handleCameraPositionChange = (axis: 'x' | 'y' | 'z', value: string) => {
    setCameraPosition((prevPosition) => ({
      ...prevPosition,
      [axis]: parseFloat(value),
    }));
  };

  
  
  const renderCrane = (scene: THREE.Scene) => {
    // get all the dimensions and dynamic states from the config object
    if (!craneState) return;
    const starting_point = craneState.position
    const orientation = craneState.orientation
    const base_dimensions = craneState.dimensions.base
    const lift_dimensions = craneState.dimensions.lift
    const upper_arm_dimensions = craneState.dimensions.upperArm
    const lower_arm_dimensions = craneState.dimensions.lowerArm
    const palm_dimensions = craneState.dimensions.palm
    const left_finger_dimensions = craneState.dimensions.leftFinger
    const right_finger_dimensions = craneState.dimensions.rightFinger

    const base_dynamic_state = craneState.dynamicState.base
    const elbow_dynamic_state = craneState.dynamicState.elbow
    const wrist_dynamic_state = craneState.dynamicState.wrist
    var lift_dynamic_state = craneState.dynamicState.lift
    var gripper_dynamic_state = craneState.dynamicState.gripper


    // Create the ground
    const ground = createCube(scene, 0x808080, { position: [0, -.5, 0], dimensions: [100, 0.1, 100] });
    console.log("starting point", orientation)
    const base_position = starting_point                          
    const base_state = {
      position: base_position,
      rotation: [0, orientation, 0],
      dimensions: base_dimensions
    }

    // create a calibration cube
    // const calibration_cubee = createCube(scene, 0xff03f0, { position: [10, 5, 10], dimensions: [0.5, 10.0, 0.5] });

    // Create the base
     const base = createCube(scene, 0x00ff00, base_state );

    
    const lift_position = [0, base_dimensions[1]/2 + lift_dimensions[1]/2, 0]
    const lift_state = {
      position: lift_position,
      rotation: [0,base_dynamic_state,0],
      dimensions: lift_dimensions
    }

    // create the lift
    const lift = createCube(base, 0x0000ff, lift_state);

    const upper_arm_position = [upper_arm_dimensions[0]/2 + lift_dimensions[0]/2, 
                                -lift_dimensions[1]/2 + upper_arm_dimensions[1]/2 + lift_dynamic_state,
                                 0]
    const upper_arm_state = {
      position: upper_arm_position,
      rotation: [0,0,0],
      dimensions: upper_arm_dimensions
    }

    // create the upper arm
    const upperArm = createCube(lift, 0xff0000, upper_arm_state);

    

    // create the lower arm

    const z_offset = (lower_arm_dimensions[0]/2 - upper_arm_dimensions[1]/2 ) * Math.sin(elbow_dynamic_state)
    const x_offset = (lower_arm_dimensions[0]/2 - upper_arm_dimensions[1]/2) *  Math.cos(elbow_dynamic_state)
    const lower_arm_position = [ +(upper_arm_dimensions[0]/2 - upper_arm_dimensions[1]/2  ) + x_offset ,
                                -lower_arm_dimensions[1] ,
                                -z_offset ]
    const lower_arm_state = {
      position: lower_arm_position,
      rotation: [0, elbow_dynamic_state, 0],
      dimensions: lower_arm_dimensions
    }
    const lowerArm = createCube(upperArm, 0xff000, lower_arm_state);

    // create the gripper composed of three cubes


    const z_offset_wrist = (palm_dimensions[0]/2 - lower_arm_dimensions[1]/2 ) * Math.sin(wrist_dynamic_state)
    const x_offset_wrist = (palm_dimensions[0]/2 - lower_arm_dimensions[1]/2) *  Math.cos(wrist_dynamic_state)
    const palm_position = [ (lower_arm_dimensions[0]/2 - lower_arm_dimensions[1]/2  ) + x_offset_wrist ,
                                -palm_dimensions[1] ,
                                -z_offset_wrist ]
    const palm_state = {
      position: palm_position,
      rotation: [0, wrist_dynamic_state, 0],
      dimensions: palm_dimensions
    }
    const palm = createCube(lowerArm, 0xff0000, palm_state ) ;

    const left_finger_position = [palm_dimensions[0]/2 - left_finger_dimensions[0]/2, -left_finger_dimensions[1]/2, 0]
    const left_finger_state = {
      position: left_finger_position,
      rotation: [0, 0, 0],
      dimensions: left_finger_dimensions
    }

    const leftFinger = createCube(palm, 0xff0000, left_finger_state ) ;

    // make sure the gripperstate is never larger than palm_dimensions[0]

    const right_finger_position = [left_finger_position[0] - gripper_dynamic_state, left_finger_position[1], left_finger_position[2]]
    const right_finger_state = left_finger_state
    right_finger_state.position = right_finger_position
    const rightFinger = createCube(palm, 0xff0000, right_finger_state ) ;

    return palm;
  }



  const createCube = (parent: THREE.Object3D , colour: number, state : { position?: number[], rotation?: number[], dimensions?: number[] }) => {
    const position = state.position || [0, 0, 0];
    const rotation = state.rotation || [0, 0, 0];
    const dimensions = state.dimensions || [1, 1, 1];

    const geometry = new THREE.BoxGeometry(dimensions[0], dimensions[1], dimensions[2]);
    const material = new THREE.MeshPhongMaterial({ color: colour });
    const cube = new THREE.Mesh(geometry, material);
    cube.rotation.set(rotation[0], rotation[1], rotation[2]);
    cube.position.set(position[0], position[1], position[2]);
    
    parent.add(cube);
    return cube;
  }

  const initThreeScene = () => {
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(
      75,
      window.innerWidth / window.innerHeight,
      0.1,
      1000
    );
    const renderer = new THREE.WebGLRenderer();
    renderer.setSize(window.innerWidth, window.innerHeight);
    
    if (canvasRef.current) {
      canvasRef.current.appendChild(renderer.domElement);
    }
    camera.position.z = cameraPosition.z;
    camera.position.y = cameraPosition.y;
    camera.position.x = cameraPosition.x;

    
    // Add a light
    const color = 0xFFFFFF;
    const intensity = 3;
    const light = new THREE.DirectionalLight(color, intensity);
    light.position.set(-1, 2, 5);
    scene.add(light);
    
    // Add ambient light
    const ambientLight = new THREE.AmbientLight(0x404040); // soft white light
    scene.add(ambientLight);
    
    // Make the background grey
    scene.background = new THREE.Color(0x808080);

    // Create a Grid
    const gridHelper = new THREE.GridHelper(110, 110); // Creates a 50x50 grid
    scene.add(gridHelper);

    // add axis helper 
    const axesHelper = new THREE.AxesHelper(15); // The parameter defines the size of the axes.

    // setcolor of the axes red for x green for y and blue for z
    axesHelper.setColors(0xff0000, 0x00ff00, 0x0000ff);
    scene.add(axesHelper);

    sceneRef.current = scene;
    
    return { scene, camera, renderer };
  };
  
  // TODO seems to be causing a lot of overhead maybe use an useEffect instead
  const animate = (camera: THREE.Camera, scene: THREE.Scene, renderer: THREE.Renderer ) => {
    const render = () => {
      requestAnimationFrame(render);
      renderer.render(scene, camera);
    };
    render();
  };

  const onResize = (camera: THREE.PerspectiveCamera, renderer: THREE.Renderer) => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  };
  

    // State to hold input values

  return (
    <Box className={styles.container} sx={{ display: 'flex' }}>
      <Box className={styles.sidebar}   sx={{ 
          flexDirection: 'column', 
          p: 2,
          height: '90vh', // Adjust the height as needed
          overflowY: 'auto' // This enables vertical scrolling
        }}>
        <Box className={styles.controlsBanner} sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
          <h2>Controls</h2>
          <IconButton aria-label="camera">
            <img src={favicon.src} alt="logo" />
          </IconButton>
        </Box>
        <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', mb: 1 }}>
          <TextField
            label="Camera X"
            type="number"
            value={metersToMm(cameraPosition.x)}
            onChange={(e) => handleCameraPositionChange('x', mmToMeters(e.target.value))}
            variant="outlined"
            size="small"
          />
          <TextField
            label="Camera Y"
            type="number"
            value={metersToMm(cameraPosition.y)}
            onChange={(e) => handleCameraPositionChange('y', mmToMeters(e.target.value))}
            variant="outlined"
            size="small"
          />
          <TextField
            label="Camera Z"
            type="number"
            value={metersToMm(cameraPosition.z)}
            onChange={(e) => handleCameraPositionChange('z', mmToMeters(e.target.value))}
            variant="outlined"
            size="small"
          />
        </Box>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1, marginTop: '10px', padding: 2, border: '1px solid', borderColor: 'divider' }}>
          <Typography variant="h8">End Effector Position</Typography>
          <Typography variant="body2">X: {metersToMm(endpoint.x)}</Typography>
          <Typography variant="body2">Y: {metersToMm(endpoint.y)}</Typography>
          <Typography variant="body2">Z: {metersToMm(endpoint.z)}</Typography>
        </Box>

        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1, marginTop: '20px' }}>
          <TextField
            label="Setpoint X"
            type="number"
            value={metersToMm(setpoint.x)}
            onChange={(e) => handleSetpointChange('x', mmToMeters(e.target.value))}
            variant="outlined"
            size="small"
          />
          <TextField
            label="Setpoint Y"
            type="number"
            value={metersToMm(setpoint.y)}
            onChange={(e) => handleSetpointChange('y', mmToMeters(e.target.value))}
            variant="outlined"
            size="small"
          />
           <TextField
            label="Setpoint Z"
            type="number"
            value={metersToMm(setpoint.z)}
            onChange={(e) => handleSetpointChange('z', mmToMeters(e.target.value))}
            variant="outlined"
            size="small"
          />         <TextField
            label="Setpoint phi"
            type="number"
            value={radToDeg(setpoint.phi)}
            onChange={(e) => handleSetpointChange('phi', degToRad(parseFloat(e.target.value)))}
            variant="outlined"
            size="small"
          />
          <Button variant="contained" color="primary" onClick={() => sendMessage(
          {
                action: "control_pid",
                data: {"setpoint" : [setpoint.x, setpoint.y,setpoint.z, setpoint.phi] } 
          }
          )}>
            Send Setpoint
          </Button>
          <Button variant="contained" color="primary" onClick={() => sendMessage(
            {
            action: "move_base",
            }
          )}>
           Move Base 
          </Button>
          <Button variant="contained" color="primary" onClick={() => sendMessage(
            {
              action: "return_to_origin",
            }
          )}>
           Return to Origin 
          </Button>
          <Button variant="contained" color="primary" onClick={() => sendMessage(
            {
            action: "move_base_and_control_pid",
            "data" : {"setpoint" : [setpoint.x, setpoint.y,setpoint.z, setpoint.phi] }
            }
          )}>
           Dance 
          </Button>
        </Box>

      </Box>

      {/* Input Box for Crane Adjustments directly above the canvas */}
    <Box sx={{ display: 'flex', flexDirection: 'column', flexGrow: 1, minWidth: 0 }}>
      {/* Inputs row directly above the canvas */}
      <Box sx={{ 
          display: 'flex', 
          justifyContent: 'space-around', // This spreads out the input fields evenly
          padding: '8px', // Adjust padding as needed
          borderBottom: '1px solid #ccc', // Optional border for visual separation
          backgroundColor: '#f5f5f5' // Optional background color for contrast
        }}>
        <TextField
          label="Base Angle"
          type="number"
          variant="outlined"
          value = {radToDeg( actuation.base)}
          onChange={(e) => 
              setActuation((prevActuation) => ({
                ...prevActuation,
                base : degToRad( parseFloat(e.target.value))
              }))
            }
          size="small"
        />
        <TextField
          label="Wrist Angle"
          type="number"
          value = {radToDeg( actuation.wrist)}
          onChange={(e) => 
              setActuation((prevActuation) => ({
                ...prevActuation,
                wrist : degToRad( parseFloat(e.target.value))
              }))
            }
          variant="outlined"
          size="small"
        />
        <TextField
          label="Elbow Angle"
          type="number"
          value = {radToDeg( actuation.elbow)}
          onChange={(e) => 
              setActuation((prevActuation) => ({
                ...prevActuation,
                elbow : degToRad( parseFloat(e.target.value))
              }))
            }
          variant="outlined"
          size="small"
        />
        <TextField
          label="Lift Height"
          type="number"
          value = {metersToMm( actuation.lift)}
          onChange={(e) => 
              setActuation((prevActuation) => ({
                ...prevActuation,
                lift: mmToMeters( parseFloat(e.target.value))
              }))
            }
          variant="outlined"
          size="small"
        />
        <TextField
          label="Gripper Opening"
          type="number"
          value = {metersToMm(actuation.gripper)}
          onChange={(e) => 
              setActuation((prevActuation) => ({
                ...prevActuation,
                gripper: mmToMeters( parseFloat(e.target.value))
              }))
            }
          variant="outlined"
          size="small"
        />
      </Box>

      <Button variant="contained" color="primary" onClick={() => sendMessage(
        {
        action: "control_pid",
        "data" : {"setpoint_actuation" : [actuation.base, actuation.elbow, actuation.wrist, actuation.lift, actuation.gripper, 0,0] }
        }
      )}>
        Send setpoint
      </Button>

      <Box className={styles.canvasContainer} ref={canvasRef} sx={{ flexGrow: 1 }}>
      </Box>
    </Box>
  </Box>
  );
}


export default Home;
