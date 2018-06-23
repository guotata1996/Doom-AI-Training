import math

linedmap = open('O:\\Doom\\scenarios\\autogen\\extracted\\TEXTMAP','r+')
addon_things = []
interest_vertex = set()

class addon_thing:
    def __init__(self):
        self.v1 = None
        self.v2 = None
        self.tid = None

if __name__ == '__main__':
    while True:
        aline = linedmap.readline()
        if not aline:
            break
        if aline.startswith('linedef'):
            playeruse = False
            playercross = False
            special = -1
            v1 = v2 = -1
            while True:
                line = linedmap.readline()
                if line.startswith('}'):
                    if special > 0:
                        assert (v1 >= 0 and v2 >= 0)
                        new_thing = addon_thing()
                        new_thing.v1 = v1
                        new_thing.v2 = v2
                        interest_vertex.add(v1)
                        interest_vertex.add(v2)

                        if special == 13:
                            new_thing.tid = 913 #key_door raise
                        elif special == 11 or 12:
                            new_thing.tid = 911 #door open/remote control
                        elif special == 70:
                            new_thing.tid = 970 #teleporter
                        elif special == 243:
                            new_thing.tid = 943 #exit
                        elif playeruse:
                            new_thing.tid = 900 #
                        elif playercross:
                            new_thing.tid = 901
                        addon_things.append(new_thing)
                    break
                if line.startswith('v1'):
                    line = line.replace(' ','')
                    line = line.replace(';','')
                    digit = line.split('=')[-1]
                    v1 = int(digit)
                if line.startswith('v2'):
                    line = line.replace(' ','')
                    line = line.replace(';','')
                    digit = line.split('=')[-1]
                    v2 = int(digit)
                if line.startswith('playercross'):
                    playercross = True
                if line.startswith('playeruse'):
                    playeruse = True
                if line.startswith('special'):
                    line = line.replace(' ','')
                    line = line.replace(';','')
                    digit = line.split('=')[-1]
                    special = int(digit)

        elif aline.startswith('vertex'):
            vertex_num = aline.split(' ')[-2]
            idx = int(vertex_num)

            if idx in interest_vertex:
                x = y = None
                while True:
                    line = linedmap.readline()
                    if line.startswith('x'):
                        line = line.replace(' ','')
                        line = line.replace(';','')
                        digit = line.split('=')[-1]
                        x = int(float(digit))
                    if line.startswith('y'):
                        line = line.replace(' ','')
                        line = line.replace(';','')
                        digit = line.split('=')[-1]
                        y = int(float(digit))
                    if line.startswith('}'):
                        assert (x is not None)
                        assert (y is not None)
                        for thing in addon_things:
                            if thing.v1 == idx:
                                thing.v1 = (x, y)
                            if thing.v2 == idx:
                                thing.v2 = (x, y)
                        break

    linedmap.seek(0, 2)
    for athing in addon_things:
        x1, y1 = athing.v1
        x2, y2 = athing.v2
        tid = athing.tid
        assert (x1 is not None) and (y1 is not None) and (x2 is not None) and (y2 is not None) and (tid is not None)
        length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        r_x = (x2 - x1) / length
        r_y = (y2 - y1) / length
        n_x = r_y
        n_y = -r_x

        dest_x = (x1 + x2) / 2 + n_x * 24
        dest_y = (y1 + y2) / 2 + n_y * 24
        dest_x = int(round(dest_x/24.0)) * 24
        dest_y = int(round(dest_y/24.0)) * 24

        thing_str = '''
thing
{{
x={}.000;
y={}.000;
type=9040;
angle=0;
skill1=true;
skill2=true;
skill3=true;
skill4=true;
skill5=true;
single=true;
coop=true;
dm=true;
id = {};
}}\n'''.format(dest_x, dest_y, tid)

        linedmap.writelines(thing_str)

    linedmap.close()