from keras.models import model_from_json
from SPECtogram import gimmeDaSPECtogram
import numpy as np
import pyaudio
import threading
import collections
import wave
from keras.utils import to_categorical
import os
import serial
import sys
import librosa

DATA_PATH = "./data/"
feature_dim_1 = 97
feature_dim_2 = 12
channel = 1
numWav = 0


json_file = open('model_3.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_3.h5")
print("Loaded model from disk")

# Input: Folder Path
# Output: Tuple (Label, Indices of the labels, one-hot encoded labels)
def get_labels(path=DATA_PATH):
    #labels = os.listdir(path)
    labels = ['up', 'down', 'left', 'right', 'one', 'two', 'three', 'four', 'stop', 'go']
    #print(labels)
    label_indices = np.arange(0, len(labels))
    return labels, label_indices, to_categorical(label_indices)


# Predicts one sample
def predict(filepath, model):
    sample = gimmeDaSPECtogram(filepath)
    print(sample.shape)
    while sample.shape[1] > 97:
        sample = sample[:,:-1].copy()
        #print(sample.shape)

    sample_reshaped = sample.reshape(1, feature_dim_1, feature_dim_2, channel)
    return get_labels()[0][
            np.argmax(model.predict(sample_reshaped))
    ]

print(predict("data/down/0ba018fc_nohash_2.wav", loaded_model))
#print(predict("MumDown.wav", loaded_model))








baudRate = 9600

testingWithArduino = False


#####################################################
####### Audio input variables
#####################################################

CHUNK: int = 1024
FORMAT: int = pyaudio.paInt16
CHANNELS: int = 2
RATE: int = 16000
RECORD_SECONDS: int = 5
WAVE_OUTPUT_FILENAME: str = "output.wav"

p = pyaudio.PyAudio()

# You can specify which microphone input device you want to use
micDeviceIndex: int = -1
RMSthreshold: int = 2000
voiceExtractTimeSeconds: float = 1
lookBackBufferLength: int = 3 #43 is a second of length
audioCutSplitChunks: int = 20
scoreThreshold: float = 0.3

info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')
for i in range(0, numdevices):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))
            if micDeviceIndex == -1:
                if p.get_device_info_by_host_api_device_index(0, i).get('name') == "default":
                    micDeviceIndex = i



stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                input_device_index=micDeviceIndex)

if stream.is_active():
    print("* recording")
else:
    print("* microphone serial connection not started")


def threadFunction(bufferInclude, loadedModel, wavNum):

    print(bufferInclude)
    listOfWavData = []
    for thing in bufferInclude:
        listOfWavData.append(thing)

    streamLocal = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    input_device_index=micDeviceIndex)

    for i in range(0, int(RATE / CHUNK * voiceExtractTimeSeconds)):
        data = streamLocal.read(CHUNK)
        listOfWavData.append(data)
        if calculateRMS(data, CHUNK) < RMSthreshold:
            break

    print("Made it past recording")

    #WAVE_OUTPUT_FILENAME = "tempWavs/tempWav" + str(threading.current_thread().ident) + ".wav"
    WAVE_OUTPUT_FILENAME = "tempWavs/tempWav" + str(wavNum) + ".wav"
    wavNum = wavNum + 1

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(listOfWavData))
    wf.close()

    print(predict(WAVE_OUTPUT_FILENAME, loadedModel))
    #os.remove(WAVE_OUTPUT_FILENAME)



def calculateRMS(data, width):

    #_checkParameters(data, width)
    if len(data) == 0: return None
    d = np.frombuffer(data, np.int16).astype(np.float)
    #print(d)
    rms = np.sqrt((d*d).sum()/len(d))
    return int(rms)





#################################################



def get_serial_port():
    if sys.platform.startswith('win'):
        # Windows platform get ports
        print("It's a windows!")
        print("Trying to get windows port automatically...")
        #ports = ['COM%s' % (i + 1) for i in range(256)]
        arduino_ports = [
            p.device
            for p in serial.tools.list_ports.comports()
            if 'Arduino' in p.description or 'CH340' in p.description
        ]
        if not arduino_ports:
            raise IOError("No Arduino found")
        if len(arduino_ports) > 1:
            print('Multiple Arduinos found - using the first')
        return arduino_ports[0]

    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        # Linux platform get ports
        print("It's a linux!")
        print("Trying to get linux port automatically...")
        ser_devs = [dev for dev in os.listdir('/dev') if dev.startswith('ttyAC') or dev.startswith('ttyUSB')]
        print(ser_devs)
        if len(ser_devs) > 0:
            return '/dev/' + ser_devs[0]
        else:
            print("No ports found")
            return None

    elif sys.platform.startswith('darwin'):
        # Mac platform get ports
        print("It's a mac!")
        print("Trying to get mac port automatically...")
        ports = list(serial.tools.list_ports.comports())


        if len(ports) == 0:
            print("No ports found")
            return None
        else:
            for p in ports:
                print(p)

            arduinoPort = ports[0]
            arduinoPortName = "/dev/" + arduinoPort.name
            return arduinoPortName

    else:
        raise EnvironmentError('Error finding ports on your operating system')



def main(loadedModel, numWav):
    bufferInclude = collections.deque(maxlen=lookBackBufferLength)
    takingDataCountdown = audioCutSplitChunks
    while 1:
        data = stream.read(CHUNK)
        bufferInclude.append(data)
        #print(calculateRMS(data, CHUNK))
        if calculateRMS(data, CHUNK) < RMSthreshold and takingDataCountdown > 0:
            takingDataCountdown -= audioCutSplitChunks
        if calculateRMS(data, CHUNK) > RMSthreshold and takingDataCountdown == 0:
            takingDataCountdown = audioCutSplitChunks
            numWav += 1
            thread = threading.Thread(target=threadFunction, args=([bufferInclude, loadedModel, numWav]))
            thread.start()

if testingWithArduino:
    port = get_serial_port()
    ser = serial.Serial(port, baudRate)

main(loaded_model, numWav)