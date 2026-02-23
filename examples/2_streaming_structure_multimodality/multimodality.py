import os
from dotenv import load_dotenv

# Clients
from datapizza.clients.google import GoogleClient
from datapizza.type import Media, MediaBlock, TextBlock

load_dotenv()


# Use multiple modalities as input like text, audio, images (depends on the model)

# Google Gemini
client = GoogleClient(
    api_key=os.getenv("GOOGLE_API_KEY"),
    model = "gemini-flash-latest",
    system_prompt = "You are a helpful AI assistant."
)

# let's pass an image as input from an URL
media = Media(
    extension="jpg",
    media_type="image",
    source_type="url",
    source="https://www.nasa.gov/wp-content/uploads/2023/03/a15pan11845-7.jpg"
)

# if want pass the image LOCALLY we need to pass it in base64 format
# import base64

# with open("image.jpg", "rb") as f:
#     image_data = base64.b64encode(f.read()).decode()


# For audio (supposing you have an audiofile)

audio_media = Media(
    extension="mp3",
    media_type="audio",
    source_type="path",
    source="path/to/audiofile.mp3",
)
# and you can ask to describe the audio or transcribe it or whatever you want



# we provide a textBlock for the text modality and a media modality for images, audio, video, etc. The model can understand both modalities and use them to generate a response.
response = client.invoke(
   [TextBlock(content="Describe the image in one sentence"), MediaBlock(media=media)] # pass image_data instead of media if you want to pass the image in base64 format (locally)
)
    
print(response.text)