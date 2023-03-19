import os

import numpy as np

sample_path = '../../../assets/urdf/soccerball.urdf'
a = 0.08
b = 0.12

def output_urdf(filename, value):
    with open(filename, 'w') as f:
        f.write("<?xml version=\"1.0\" ?>\n")
        f.write("<robot name=\"ball\">\n")
        f.write("\t<link name=\"ball\">\n")
        f.write("\t\t<inertial>\n")
        f.write("\t\t\t<origin rpy=\"0 0 0\" xyz=\"0 0 0\"/>\n")
        f.write("\t\t\t<mass value=\"0.37\"/>\n")
        f.write("\t\t\t<inertia ixx=\"0.064\" ixy=\"0.\" ixz=\"0.\" iyy=\"0.064\" iyz=\"0.0\" izz=\"0.064\"/>\n")
        f.write("\t\t</inertial>\n")

        f.write("\t\t<visual>\n")
        f.write("\t\t\t<origin rpy=\"0 0 0\" xyz=\"0 0 0\"/>\n")
        f.write("\t\t\t<geometry>\n")
        f.write("\t\t\t\t<sphere radius=\"" + str(value)[0:6] + "\"/>\n")
        f.write("\t\t\t</geometry>\n")
        f.write("\t\t\t<material name=\"mat\">\n")
        f.write("\t\t\t\t<color rgba=\"0.957 0.898 0.340 1.\"/>\n")
        f.write("\t\t\t</material>\n")
        f.write("\t\t</visual>\n")
        f.write("\t\t<collision>\n")
        f.write("\t\t\t<origin rpy=\"0 0 0\" xyz=\"0 0 0\"/>\n")
        f.write("\t\t\t<geometry>\n")
        f.write("\t\t\t\t<sphere radius=\"" + str(value)[0:6] + "\"/>\n")
        f.write("\t\t\t</geometry>\n")
        f.write("\t\t</collision>\n")

        f.write("\t</link>\n")
        f.write("</robot>\n")

    return

def main():
    os.mkdir('soccerball_urdfs')
    for i in range(4096):
        val = np.random.uniform(a, b)
        filename = "soccerball_urdfs/" + str(val)[0] + str(val)[2:6] + '_' + str(i) + ".urdf"
        output_urdf(filename, val)
    return

if __name__ == "__main__":
    main()
