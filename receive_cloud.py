from azure.iot.device import IoTHubDeviceClient, Message
from threading import Thread
import json
import os
class Recieve_C2D_Message():
    def __init__(self,cs):
        self.cs = cs
        self.client = IoTHubDeviceClient.create_from_connection_string(self.cs)
        self.message = None
        self.camera_source = None

    def receive_message(self):

        while True:
            self.message = self.client.receive_message()
            self.camera_source = str(self.message.data).split('b')[1].split("'")[1]
            print("\nMessage received:")
            print("Data: {}".format(str(self.message.data)))
            temp_source = str(self.message.data).split('b')[1].split("'")[1]
            source = {"source":temp_source}
            with open('video_source.json', 'w') as outfile:
                json.dump(source, outfile)
            print(self.message.data)

        while True:
            time.sleep(1000)

    def start(self):
        self.thread = Thread(target = self.receive_message, args = ())
        #sself.thread.daemon = True    
        self.thread.start()


if __name__ == '__main__':
    receive_ = Recieve_C2D_Message(cs=os.getenv("cs"))
    receive_.start()
