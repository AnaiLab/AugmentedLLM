import os
from striprtf.striprtf import rtf_to_text
from config import Config

# Edit the following code if you don't want to use the config file
rtf_file_dir = Config.RTF_FILE_DIR
converted_file_dir = Config.CONVERTED_FILE_DIR

directory = os.fsencode(rtf_file_dir)    
for file in os.listdir(directory):
     filename = os.fsdecode(file)
     if filename.endswith(".rtf"): 

        # read in data
        f = open(rtf_file_dir + filename, 'r')
        rtf = f.read()
        f.close()

        txt = rtf_to_text(rtf)

        # write out data
        f = open(converted_file_dir + filename.replace(".rtf", ".txt"), 'w')
        f.write(txt)
        f.close()