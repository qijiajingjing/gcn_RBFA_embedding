import numpy as np
import pandas as pd
import os
'''
Generating nodes dict.
'''
poscar_path = '../DD/'
poscar_path_slab = '../DD-slab/'
element_df=pd.read_excel('./element_info_ND.xlsx')

'''make label(G_OH) from DFT data'''
G_H2 = -6.902
G_H2O = -14.254
G_O2 = -9.784

read_row_label = open('DFT_data.txt')
file_content = read_row_label.readlines()
data_list = []
for i in range(len(file_content)):
    tt = []
    for j in file_content[i].split():
        tt.append(j)
    data_list.append(tt)
G_df = pd.DataFrame(data_list[:], columns=['Component', 'Distance_type', 'site', 'E_DFT', 'ZEP', 'TS', 'slab'])
G_df.iloc[:, 3] = G_df.iloc[:, 3].astype('float')  # E_DFT
G_df.iloc[:, 4] = G_df.iloc[:, 4].astype('float')  # E_zep
G_df.iloc[:, 5] = G_df.iloc[:, 5].astype('float')  # E_TS
G_df.iloc[:, 6] = G_df.iloc[:, 6].astype('float')  # E_slab_DFT

G = G_df.iloc[:, 3] + G_df.iloc[:, 4] - G_df.iloc[:, 5]  # G_OH
E_slab = G_df.iloc[:, 6]
G_df.insert(G_df.shape[1], 'G', G)
G_OH = G - E_slab - (G_H2O - 0.5 * G_H2)
G_df.insert(G_df.shape[1], 'G_OH', G_OH)
G_df.to_csv('./Free_energy_Dataframe.csv', index=False, sep=',')

def get_element_feature(element):
    for i in np.arange(len(element_df)):
        if element == element_df.iloc[i][0]:
            a=list(np.array(element_df.iloc[i,1:]))
    return a

def calculate_distance(XYZ1,XYZ2):
    d=((XYZ1[0]-XYZ2[0])**2+(XYZ1[1]-XYZ2[1])**2+(XYZ1[2]-XYZ2[2])**2)**0.5
    return d

def make_data():
    poscar_list = os.listdir(poscar_path_slab)
  # poscar_list = poscar_list[:10]
    NG = len(poscar_list)
    graph_indictor = []
    node_feature = []
    mark_node = []
    for i in np.arange(NG):
        graph_indictor.append([poscar_list[i],i])
    for i in np.arange(NG):
        print(i,poscar_list[i])
        element_mark=[]
        f=open(poscar_path_slab + poscar_list[i])
        f_content=[]
        for line in f.readlines():
            line = line.split()
            if line == []:
                break
            f_content.append(line)
        f.close()
        element_mark.append(f_content[5])
        element_mark.append(np.array([int(x) for x in f_content[6]]).cumsum())
        for j in np.arange(8,len(f_content)):
            t = 0
            for k in np.arange(len(element_mark[0])-1):
                if j-7 > element_mark[1][k]  and j-7 <= element_mark[1][k+1]:
                    if element_mark[0][k+1] != 'C':
                        mark_node.append(True)
                    else:
                        mark_node.append(False)
                    node_feature.append([i,[float(x) for x in f_content[j][:3]],get_element_feature(element_mark[0][k+1])])
                    break
                else:
                    t=t+1
            if t==len(element_mark[0])-1:
                if element_mark[0][0] != 'C':
                    mark_node.append(True)
                else:
                    mark_node.append(False)
                node_feature.append([i,[float(x) for x in f_content[j][:3]],get_element_feature(element_mark[0][0])])
    distance_threshold = 2.1
    DD_A = []
    for i in np.arange(NG):
        print(i,poscar_list[i])
        select_atom=[]
        for j in np.arange(len(node_feature)):
            if node_feature[j][0]==i:
                select_atom.append([j,node_feature[j][1]])
        for j in np.arange(len(select_atom)):
            for k in np.arange(len(select_atom)):
                if calculate_distance(select_atom[j][1],select_atom[k][1]) <= distance_threshold and calculate_distance(select_atom[j][1],select_atom[k][1]) > 0:
                    DD_A.append([select_atom[j][0],select_atom[k][0]])
    poscarlist=open(poscar_path+'/poscarlist.txt','w+')
    for i in range(len(poscar_list)):
        poscarlist.write(str(i)+' '+poscar_list[i])
        poscarlist.write('\n')
    poscarlist.close()
    return node_feature, graph_indictor, DD_A, mark_node

node_feature , graph_indictor , DD_A, mark_node=make_data()

# _1 means contcar
DD_A_f=open(poscar_path+'/DD_A.txt', 'w+')
for i in np.arange(len(DD_A)):
    DD_A_f.write(str(DD_A[i][0])+' , '+str(DD_A[i][1]))
    DD_A_f.write('\n')
DD_A_f.close()

DD_graph_indicator=open(poscar_path+'DD_graph_indicator.txt', 'w+')
for i in np.arange(len(node_feature)):
    DD_graph_indicator.write(str(node_feature[i][0])+'\n')
DD_graph_indicator.close()

DD_node_labels=open(poscar_path+'DD_node_labels.txt', 'w+')
for i in np.arange(len(node_feature)):
    for j in np.arange(len(node_feature[i][2])):
        DD_node_labels.write(str(node_feature[i][2][j])+' ')
    DD_node_labels.write('\n')
DD_node_labels.close()

DD_graph_label=open(poscar_path+'DD_graph_labels.txt', 'w+')
for i in np.arange(len(graph_indictor)):
    a=graph_indictor[i][0].split('+')
    y2=G_df[(G_df.Component==a[0])&(G_df.Distance_type==a[1])&(G_df.site==a[2])].iloc[0][8]
    DD_graph_label.write(str(y2)+'\n')
    #DD_graph_label.write(str(y1)+' '+str(y2)+'\n')
DD_graph_label.close()

mark_node_f=open(poscar_path+'DD_atten_scores.txt','w+')
for i in range(len(mark_node)):
    mark_node_f.write(str(mark_node[i]))
    mark_node_f.write('\n')
mark_node_f.close()


