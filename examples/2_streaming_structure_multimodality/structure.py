import os
from dotenv import load_dotenv

# Clients
from datapizza.clients.google import GoogleClient

# Pydantic base model
from pydantic import BaseModel

load_dotenv()


# Make AI agent to work with structured data i.e. JSON and tables.
# We leverage pydantic to define the structure of the data and datapizza will take care of parsing it and validating it.
# In this way we create strictured objects that the model can use extract information as input or assign different fields in an output.


# Google Gemini
client = GoogleClient(
    api_key=os.getenv("GOOGLE_API_KEY"),
    model = "gemini-flash-latest",
    system_prompt = "You are a helpful AI assistant."
)


# Pydantic base model

class Actor(BaseModel):
    name: str
    age: int
    movies: list[str]
    
class MovieDescription(BaseModel):
    title: str
    year: int
    genre: str
    actors: list[Actor]
    

prompt = """
Extract the information about the movie from the following description and return it in a structured format as a JSON that follows the MovieDescription structure.
Movie description: "Inception is a 2010 science fiction action film directed by Christopher Nolan. The movie stars Leonardo DiCaprio as Dom Cobb, a skilled thief who steals secrets from deep within the subconscious during the dream state. The cast also includes Joseph Gordon-Levitt as Arthur, Ellen Page as Ariadne, and Tom Hardy as Eames. The film explores themes of dreams, reality, and the nature of consciousness."
"""

response = client.structured_response(input=prompt, output_cls=MovieDescription)
    
movie_description = response.structured_data[0] # we can have multiple structured objects in the response

print(f"Title: {movie_description.title}") # Inception
print(f"Year: {movie_description.year}") # 2010
print(f"Genre: {movie_description.genre}") # science fiction action
print(f"Actor Name: {movie_description.actors[0].name}") # Leonardo DiCaprio
print(f"Actor Age: {movie_description.actors[0].age}") # We didn't provide the age in the prompt, but the model can still assign a value to it based on its knowledge.
print(f"Actor Movies: {movie_description.actors[0].movies}") # We didn't provide the movies in the prompt, but the model can still assign a value to it based on its knowledge.