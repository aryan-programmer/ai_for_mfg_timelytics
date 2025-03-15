import os


def pickle_split(source, dest_folder, write_size):
    # Make a destination folder if it doesn't exist yet
    if not os.path.exists(dest_folder):
        os.mkdir(dest_folder)
    else:
        # Otherwise clean out all files in the destination folder
        for file in os.listdir(dest_folder):
            os.remove(os.path.join(dest_folder, file))
    partnum = 0

    # Open the source file in binary mode
    input_file = open(source, 'rb')
    while True:
        # Read a portion of the input file
        chunk = input_file.read(write_size)

        # End the loop if we have hit EOF
        if not chunk:
            break

        # Increment partnum
        partnum += 1

        # Create a new file name
        filename = dest_folder + 'voting_model' + str(partnum)

        # Create a destination file
        dest_file = open(filename, 'wb')

        # Write to this portion of the destination file
        dest_file.write(chunk)        # Explicitly close
        dest_file.close()    # Explicitly close
    input_file.close()    # Return the number of files created by the split
    return partnum


pickle_split(source='./voting_model.pkl', write_size=20000000, dest_folder='./ModelFiles/')
