import os
import gdown
import time

def download_data_folder(id):
    url = f"https://drive.google.com/file/d/{id}/view?usp=sharing"
    output = "data.zip"
    gdown.download(url, output, fuzzy=True)
    
    
if __name__=="__main__":
    download_data_folder("1UXV11s_SoAMO9wDB8Wpfgz-5pAphdEtn")
    os.system("unzip data.zip")
    os.system("rm data.zip")
    os.system("mkdir data/webqsp/lf_integrator")
    os.system("mkdir data/webqsp/sketch_generation")
    os.system("mkdir data/grailqa/lf_integrator")
    os.system("mkdir data/grailqa/sketch_generation")
    os.system("mkdir data/grailqability/a/lf_integrator")
    os.system("mkdir data/grailqability/a/sketch_generation")
    os.system("mkdir data/grailqability/au/lf_integrator")
    os.system("mkdir data/grailqability/au/sketch_generation")
    