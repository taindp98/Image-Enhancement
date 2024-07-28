# Check if the .env file exists
if [ ! -f .env ]
then
  # If the .env file exists, export the environment variables defined in it
  export $(cat .env | xargs)
fi

# Set the working directory to the current directory
export WORKDIR=$(pwd)
# Add the working directory to the PYTHONPATH
export PYTHONPATH="$WORKDIR:$PYTHONPATH"
python image_enhancement/denoise.py