from robot_connection.client_automated import send_path

class Communicator:
    def __init__(self, config):
        self.ip=config.ROBOT_IP
        self.port=config.ROBOT_PORT

    def send(self, path, heading):
        if path:
            send_path(self.ip,self.port,path,heading)