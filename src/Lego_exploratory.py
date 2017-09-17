import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from IPython.display import Image
from collections import Counter
import operator, random


'''
Capstone project for Galvanize Data Science Immersion course, Seattle
Brian Silverstein, 2017
Ben's Bricks BINS TO BRICKS App

This file will take functions currently in Lego_Exploratory_Data_Analysis.ipynb
'''

def read_all_rebrickable_files(curr_dir):
    '''Import all data files from the Rebrickable database
       INPUT:  path name of directory with data files
       OUTPUT: tuple of pandas dataframes containing everything:
                    colors,inventories,inventory_parts
                    inventory_sets, part_categories,
                    parts,sets,themes
    '''
    colors = pd.read_csv(curr_dir + 'colors.csv')
    inventories = pd.read_csv(curr_dir + 'inventories.csv')
    inventory_parts = pd.read_csv(curr_dir + 'inventory_parts.csv')
    inventory_sets = pd.read_csv(curr_dir + 'inventory_sets.csv')
    part_categories = pd.read_csv(curr_dir + 'part_categories.csv')
    parts = pd.read_csv(curr_dir + 'parts.csv')
    sets = pd.read_csv(curr_dir + 'sets.csv')
    themes = pd.read_csv(curr_dir + 'themes.csv')

    return (colors,inventories,inventory_parts,inventory_sets,
            part_categories,parts,sets,themes)

def set_id_from_kit_name(lego_set,inventories):
    '''converts between two notations used in Rebrickable database, for
    the names/numbers associated with Lego sets, taking the set ID (string)
    and converting to the invertory ID (int).
    INPUT:  lego_set (string): set_num in inventory_sets, sets, and inventories
    OUTPUT: inventory_id (int): in inventory_sets, inventory_parts
    '''
    inventory_set = inventories[inventories['set_num']==lego_set]
    inventory_id = np.max(inventory_set['id'])
    return inventory_id


def lists_from_inv_id(lego_set, occ,inventories,inventory_parts,missing=1.0):
    '''This function is part of the tf-idf process.  It takes in a lego set_num
       name (string) and a dictionary "bag of words" of parts in a collection
       and returns some information on their relationship.
       INPUTS:  lego_set (string): set_num in inventory_sets, sets, and inventories
                occ (dict): all part numbers in bag of words, counted
       OUTPUTS: this_set_occ (dict): part numbers in this lego set, counted
                this_set_term_freq (dict): part numbers in this lego set, frequncy
                this_set_parts (pandas DataFrame): filtered subset of inventory_parts
                inventory_id (int): in inventory_sets, inventory_parts
    '''
    inventory_id = set_id_from_kit_name(lego_set,inventories)

    this_set_parts = inventory_parts[inventory_parts['inventory_id']==inventory_id]

    this_set_occ={}
    this_set_term_freq={}
    this_set_piece_count=1.0*sum(this_set_parts['quantity'])

    for part_id,part_count in zip(this_set_parts['part_num'],this_set_parts['quantity']):
        if random.random() < missing:
            occ[part_id]= occ.get(part_id,0)+part_count
            this_set_occ[part_id]= this_set_occ.get(part_id,0)+part_count
            this_set_term_freq[part_id]=this_set_occ[part_id]/this_set_piece_count

    return this_set_occ,this_set_term_freq,this_set_parts,inventory_id


def print_kit_summary(lego_set,inventories,sets,inventory_parts):
    '''Prints some information about the bricks in a kit
       INPUTS: lego_set (string): set_num in inventory_sets, sets, & inventories
       OUTPUT: print to screen, returns None
    '''
    inventory_id = set_id_from_kit_name(lego_set,inventories)
    print "Kit number=", lego_set
    print "ID number =", inventory_id
    print "part count=",np.max(sets[sets['set_num']==lego_set]['num_parts'])
    print "Individual parts =", len(
        inventory_parts[inventory_parts['inventory_id']==inventory_id])
    print "total parts =", sum(
        inventory_parts[inventory_parts['inventory_id']==inventory_id]['quantity'])
    print
    return None

def brick_dictionary_from_csv(info_dir,name_list,reverse=False):
    '''Reads the list of lego csv files made through Lego Digital Designer
       In order to make a dictionary converting between Lego ID formats
       Note: Just changed dictionary from one of integers to one of strings
       and this probably breaks some code
    '''
    bric_dict={}
    for name in name_list:
        fname = info_dir + name
        model=pd.read_csv(fname)
        model=model[model['Brick']!='Total:']
        if reverse:
            for Brick,Part in zip(model['Brick'],model['Part']):
                #bric_dict[int(Part)]=int(Brick)
                bric_dict[int(Part)]=Brick
            bric_dict[-1]='-1'
        else:
            for Brick,Part in zip(model['Brick'],model['Part']):
                #bric_dict[int(Brick)]=int(Part)
                bric_dict[Brick]=int(Part)
            bric_dict['-1']=-1
    return bric_dict

def category_from_name(part_num,parts):
    '''This function just does a dictionary check for lookup of
       part data. Used only in Lego_Exploratory_Data_Analysis.ipynb
       INPUTS: part_num: dictionary of ID numbers
               parts: pandas df from rebrickable.com data
       OUTPUTS:part ID from dictionary
    '''
    temp=parts[parts['part_num']==part_num]
    if temp.shape[0]> 0:
        return max(temp['part_cat_id'])
    else:
        return -1

def make_y_categories(info_dir,name_list,passed_list):
    '''Takes info from rebrickable.com data and makes a list
       which I was considering using as category labels before
       deciding to go with individual lego shape ID numbers.
       Used only in Lego_Exploratory_Data_Analysis.ipynb
    '''
    all_y_stuff = pd.DataFrame(columns=['Brick','Name','Part',
                                        'csv_file_name','csv_file_num',
                                        'part_cat_id'])
    for name in name_list:
        fname = info_dir + name
        model=pd.read_csv(fname)
        model=model[model['Brick']!='Total:']
        model = model[model['Brick'].isin(passed_list)]

        model.Part = model.Part.astype(int)
        model['csv_file_name'] = name
        model['csv_file_num'] = int(name[-6:-4])
        model.drop(['Picture','Quantity','Color code'],axis=1,inplace=True)
        all_y_stuff = pd.concat([all_y_stuff,model])
    return all_y_stuff


def count_most_popular_kits(inventory_parts,bric_dict,n=5):
    '''Takes info from rebrickable.com data and makes a list
       of lego part IDs which occur in the most kits
       Used only in Lego_Exploratory_Data_Analysis.ipynb
    '''
    part_kit_count = {}
    for key in inventory_parts['part_num']:
        part_kit_count[key] = part_kit_count.get(key,0)+1
    sorted_part_kit_count = sorted(part_kit_count.iteritems(),
                                   key=operator.itemgetter(1), reverse=True)
    for row in sorted_part_kit_count[:n]:
        try:
            ID = bric_dict.get(int(row[0]),0)
        except ValueError:
            ID = 0
        print "Part shape = {:>6}, # of kits = {:<6,}, and Lego ID = {}".format(
                #row[0],row[1],bric_dict.get(int(row[0]),0))
                row[0],row[1],ID)


def count_most_popular_count(inventory_parts,bric_dict,n=5):
    '''Takes info from rebrickable.com data and makes a list
       of lego part IDs which occur MOST in a set of all kits.
       Different from "count_most_popular_kits" in that a brick
       used 100x in a kit gets a boost here, but is counted 1x there.
       Used only in Lego_Exploratory_Data_Analysis.ipynb
    '''
    part_pop_count ={}
    for key,num in zip(inventory_parts['part_num'],inventory_parts['quantity']):
        part_pop_count[key] = part_pop_count.get(key,0)+num
    sorted_part_pop_count = sorted(part_pop_count.iteritems(),key=operator.itemgetter(1), reverse=True)
    for row in sorted_part_pop_count[:n]:
        try:
            ID = bric_dict.get(int(row[0]),0)
        except ValueError:
            ID = 0
        print "Part shape = {:>6}, piece counts = {:<6,}, and Lego ID = {}".format(
        row[0],row[1],ID)



def cos_similarity(a,b):
    '''Not currently used, but will be when comparing lists to kit lists.
       INPUTS: two vectors
       OUTPUT: cosine similarity scaled so parallel=1,anti-parallel=0,perpendicular=.5
    '''
    return 0.5 + 0.5*np.dot(a,b)/np.linalg.norm(a)/np.linalg.norm(b)


if __name__ == '__main__':
    # open some data files from Rebrickable.com
    curr_dir = '~/Documents/Galvanize/DSI_class/CAPSTONE/lego-database/rebrickable data/2017-08/'
    (colors,inventories,inventory_parts,inventory_sets,
     part_categories,parts,sets,themes) = read_all_rebrickable_files(curr_dir)
