import numpy as np
TRAINING_DATA = 'wifi_data.txt'

def get_x_y_data():
    bssid_cnt={}
    place_set = set()
    f = open(TRAINING_DATA)
    content = f.readlines()
    x_data,y_data=[],[]
    zone_int_dict = {}
    start_int = 0

    for row in content:
        first_colon_index = row.find(':')
        data = eval(row[first_colon_index+1:])
        for k in data.keys():
            if k not in bssid_cnt:
                bssid_cnt[k]=0
            bssid_cnt[k]+=1


    import operator
    sorted_result = sorted(bssid_cnt.items(),key=operator.itemgetter(1))
    print 'sort keys',sorted_result
    bssid_list = [x[0] for x in sorted_result[-3:]]

    print 'bssid_list',bssid_list # use the 3 most common bssid to build the map


    for row in content:
        first_colon_index = row.find(':')
        place = row[:first_colon_index]
        place_set.add(place)
        data = eval(row[first_colon_index+1:])

        for k in data.keys():
            if k not in bssid_list:
                del(data[k])
        for bssid in bssid_list:
            if bssid not in data:
                data[bssid]='-100'

        #print place,data

        tmp_x_data = [int(data[bssid]) for bssid in bssid_list]
        if place not in zone_int_dict:
            zone_int_dict[place] = start_int
            start_int +=1
        tmp_y_data = zone_int_dict[place]
        y_data.append(tmp_y_data)
        x_data.append(tmp_x_data)
        #print place,tmp_x_data


    x_data = np.array(x_data).astype(dtype='float64')
    y_data = np.array(y_data)
    return x_data,y_data,len(place_set),zone_int_dict