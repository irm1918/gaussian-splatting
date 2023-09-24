#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import traceback
import socket
import json
from scene.cameras import MiniCam

host = "127.0.0.1"
port = 6009

conn = None
addr = None

listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

def init(wish_host, wish_port):
    """
    Initializes the socket listener with the desired host and port.
    
    Args:
        wish_host (str): The desired host for the socket listener.
        wish_port (int): The desired port for the socket listener.
    """
    global host, port, listener
    host = wish_host
    port = wish_port
    listener.bind((host, port))
    listener.listen()
    listener.settimeout(0)

def try_connect():
    """
    Tries to establish a connection with the socket listener. If successful, it sets the connection and
    address globally and prints the address of the connected client.
    """
    global conn, addr, listener
    try:
        conn, addr = listener.accept()
        print(f"\nConnected by {addr}")
        conn.settimeout(None)
    except Exception as inst:
        pass
            
def read():
    """
    Reads a message from the connection. The message is expected to be a JSON object sent as a string.
    The function first reads the length of the message, then the message itself, and finally decodes and
    parses the message into a Python dictionary.
    
    Returns:
        dict: The parsed message received from the connection.
    """
    global conn
    messageLength = conn.recv(4)
    messageLength = int.from_bytes(messageLength, 'little')
    message = conn.recv(messageLength)
    return json.loads(message.decode("utf-8"))

def send(message_bytes, verify):
    """
    Sends a message to the connection. The message is expected to be a bytes object. If the message is not
    None, it is sent first, followed by the length of the verification string and the verification string
    itself.
    
    Args:
        message_bytes (bytes): The message to be sent to the connection.
        verify (str): The verification string to be sent after the message.
    """
    global conn
    if message_bytes != None:
        conn.sendall(message_bytes)
    conn.sendall(len(verify).to_bytes(4, 'little'))
    conn.sendall(bytes(verify, 'ascii'))

def receive():
    """
    Receives a message from the connection and parses it into several variables. The message is expected
    to contain information about the resolution, training flag, field of view, near and far clipping planes,
    flags for SHS and rotation scaling in Python, keep-alive flag, scaling modifier, view matrix, and view
    projection matrix. If the resolution is not zero, a MiniCam object is created with the parsed information
    and returned along with the parsed flags and scaling modifier.
    
    Returns:
        MiniCam, bool, bool, bool, bool, float: The created MiniCam object, training flag, SHS flag, rotation
        scaling flag, keep-alive flag, and scaling modifier.
    """
    message = read()

    width = message["resolution_x"]
    height = message["resolution_y"]

    if width != 0 and height != 0:
        try:
            do_training = bool(message["train"])
            fovy = message["fov_y"]
            fovx = message["fov_x"]
            znear = message["z_near"]
            zfar = message["z_far"]
            do_shs_python = bool(message["shs_python"])
            do_rot_scale_python = bool(message["rot_scale_python"])
            keep_alive = bool(message["keep_alive"])
            scaling_modifier = message["scaling_modifier"]
            world_view_transform = torch.reshape(torch.tensor(message["view_matrix"]), (4, 4)).cuda()
            world_view_transform[:,1] = -world_view_transform[:,1]
            world_view_transform[:,2] = -world_view_transform[:,2]
            full_proj_transform = torch.reshape(torch.tensor(message["view_projection_matrix"]), (4, 4)).cuda()
            full_proj_transform[:,1] = -full_proj_transform[:,1]
            custom_cam = MiniCam(width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform)
        except Exception as e:
            print("")
            traceback.print_exc()
            raise e
        return custom_cam, do_training, do_shs_python, do_rot_scale_python, keep_alive, scaling_modifier
    else:
        return None, None, None, None, None, None