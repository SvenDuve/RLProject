25 episodes on model training

# Open a file to store the output
output_file = open("output.txt", "w")

# Redirect the stdout to the output file
redirect_stdout(output_file)

# Perform some computation
modelEnv("Pendulum-v1", ModelParameter(training_episodes=50, trajectory=10, store_frequency=5))

# Close the output file
close(output_file)


redirect_stdout(f::Function, stream)


# Create an IOBuffer to store the output
output_buffer = IOBuffer()

# Redirect the stdout to the output buffer
redirect_stdout(output_buffer)

# Perform some computation
println("Hello, world!")

# Get the output as a string
output_string = String(takebuf_string(output_buffer))

# Print the output string
println(output_string)


# Some handling of the local Gym 

- git fork the original Gymnasium
- clone to local
- apply the changes, perhaps start with some print statemtents
- git push to github

Once happy got to julia

```using Conda, PyCall```

Then

```Conda.pip("install", "git+https://github.com/SvenDuve/ThePackage")```

Check if everything is as required, if not, apply some more changes, once happy again, git push again, but then:

```Conda.pip("install --upgrade --force-reinstall", "git+https://github.com/SvenDuve/ThePackage")```