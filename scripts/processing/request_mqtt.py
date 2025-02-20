import paho.mqtt.client as mqtt

def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    client.subscribe("shelly/switch-1/status")

def on_message(client, userdata, msg):
    print(f"Topic: {msg.topic}\nMessage: {msg.payload.decode()}")

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect("MQTT_BROKER_IP", 1883, 60)
client.loop_forever()
