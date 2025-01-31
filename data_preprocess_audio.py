import pretty_midi
import numpy as np
import glob
import pandas as pd
import scipy
from scipy import signal
import pickle
import torchvision
import librosa
import os

motion_sf = 250
change_fps = 40


def flatten(l):
    return [item for sublist in l for item in sublist]


def read_midi(filename, specific_fps):
    # Load the MIDI file
    midi_data = pretty_midi.PrettyMIDI(filename)

    piano_roll = midi_data.get_piano_roll(fs=specific_fps)  # 40fps #250fps
    piano_roll[piano_roll > 0] = 1

    return piano_roll

def audio_preprocess(audio_path, specific_fps):
    n_fft = 4096
    hop = int(44000/specific_fps) #1102.5 -> 40fps #882 -> 50fps
    y, sr = librosa.load(audio_path, sr=44000) #44000 for divide 40
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_mfcc=13)
    y = np.where(y == 0, 1e-10, y)
    energy = np.log(librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop, center=True))
    mfcc_energy = np.vstack((mfcc, energy))
    mfcc_delta = librosa.feature.delta(mfcc_energy)
    aud = np.vstack((mfcc_energy, mfcc_delta)).T
    
    print("hop:", hop)
    print("aud:", aud.shape)
    return aud

# this_performer = "vio01"
performaer_list = ["vio01", "vio02", "vio03", "vio04", "vio05"]
# performaer_list = ["vio01", "vio02", "vio03", "vio04", "vio05"]
piece_list = ["Bach1_S1_T1", "Bach1_S1_T2", "Bach2_S1_T1", "Bach2_S1_T2", "Bach2_S2_T1", "Bach2_S2_T2", "Beeth1_S1_T1", "Beeth1_S1_T2", "Beeth2_S1_T1", "Beeth2_S1_T2", "Elgar_S1_T1",
              "Elgar_S1_T2", "Flower_S1_T1", "Flower_S1_T2", "Mend_S1_T1", "Mend_S1_T2", "Mozart1_S1_T1", "Mozart1_S1_T2", "Mozart2_S1_T1", "Mozart2_S1_T2", "Wind_S1_T1", "Wind_S1_T2"]

# Read Motion
keypoint_dictionary = {}
for this_performer in performaer_list:
    keypoint_dictionary[this_performer] = {}

training_dataset = {}  # 'aud', 'keypoints', 'seq_len'=900
UoE_mocap_path = "./UoE_violin_midi/performance_motion/"

body_joint_dict = {"RFHD_X": 0, "RFHD_Y": 1, "RFHD_Z": 2,  # head
                   "LFHD_X": 3, "LFHD_Y": 4, "LFHD_Z": 5,
                   "RBHD_X": 6, "RBHD_Y": 7, "RBHD_Z": 8,
                   "LBHD_X": 9, "LBHD_Y": 10, "LBHD_Z": 11,
                   "C7_X": 12, "C7_Y": 13, "C7_Z": 14,
                   "T10_X": 15, "T10_Y": 16, "T10_Z": 17,
                   "CLAV_X": 18, "CLAV_Y": 19, "CLAV_Z": 20,
                   "STRN_X": 21, "STRN_Y": 22, "STRN_Z": 23,
                   "RSHO_X": 24, "RSHO_Y": 25, "RSHO_Z": 26,
                   "RELB_X": 27, "RELB_Y": 28, "RELB_Z": 29,
                   "RWRA_X": 30, "RWRA_Y": 31, "RWRA_Z": 32,
                   "RWRB_X": 33, "RWRB_Y": 34, "RWRB_Z": 35,
                   "RFIN_X": 36, "RFIN_Y": 37, "RFIN_Z": 38,
                   "LSHO_X": 39, "LSHO_Y": 40, "LSHO_Z": 41,
                   "LELB_X": 42, "LELB_Y": 43, "LELB_Z": 44,
                   "LWRA_X": 45, "LWRA_Y": 46, "LWRA_Z": 47,
                   "LWRB_X": 48, "LWRB_Y": 49, "LWRB_Z": 50,
                   "LFIN_X": 51, "LFIN_Y": 52, "LFIN_Z": 53,
                   # (18*3) left waist
                   "RASI_X": 54, "RASI_Y": 55, "RASI_Z": 56,
                   # (19*3) right waist
                   "LASI_X": 57, "LASI_Y": 58, "LASI_Z": 59,
                   "RPSI_X": 60, "RPSI_Y": 61, "RPSI_Z": 62,
                   "LPSI_X": 63, "LPSI_Y": 64, "LPSI_Z": 65,
                   "RKNE_X": 66, "RKNE_Y": 67, "RKNE_Z": 68,
                   "RHEE_X": 69, "RHEE_Y": 70, "RHEE_Z": 71,
                   "RTOE_X": 72, "RTOE_Y": 73, "RTOE_Z": 74,
                   "RANK_X": 75, "RANK_Y": 76, "RANK_Z": 77,
                   "LKNE_X": 78, "LKNE_Y": 79, "LKNE_Z": 80,
                   "LHEE_X": 81, "LHEE_Y": 82, "LHEE_Z": 83,
                   "LTOE_X": 84, "LTOE_Y": 85, "LTOE_Z": 86,
                   "LANK_X": 87, "LANK_Y": 88, "LANK_Z": 89,  # 29*3
                   "Rightant_X": 90, "Rightant_Y": 91, "Rightant_Z": 92,
                   "Rightpost_X": 93, "Rightpost_Y": 94, "Rightpost_Z": 95,
                   "Leftant_X": 96, "Leftant_Y": 97, "Leftant_Z": 98,
                   "Leftpost_X": 99, "Leftpost_Y": 100, "Leftpost_Z": 101}

shape_dict = {}

shape_dict['selected_frame_shape'] = []
shape_dict['combined_shape'] = []
keypoint_list = []
target_dict = {}

for this_performer in performaer_list:
    for song_name in piece_list:
        file_full_path = UoE_mocap_path + this_performer + \
            "/" + this_performer + "_" + song_name + ".csv"
        print("===============================")
        print(file_full_path)
        splited_name = str(file_full_path).split('/')
        name_code = splited_name[4].split('.')[0]
        print("\nthis performer: ", this_performer)
        print("\nname_code:", name_code)

        mocap_metadata = pd.read_csv(file_full_path, nrows=5, names=[
                                     'attr', 'value'], on_bad_lines='skip', header=None)
        mocap_data = pd.read_csv(
            file_full_path, skiprows=5, on_bad_lines='skip')

        mocap_n_frame = mocap_metadata.loc[mocap_metadata['attr']
                                           == "NO_OF_FRAMES", "value"].item()
        mocap_fps = mocap_metadata.loc[mocap_metadata['attr']
                                       == "SAMPLING FREQUENCY (FPS)", "value"].item()
        mocap_n_frame = int(mocap_n_frame)
        mocap_fps = int(mocap_fps)
        print("mocap_n_frame", mocap_n_frame)
        print("mocap_fps", mocap_fps)
        print(mocap_data.shape)

        # Fill NAN
        mocap_data = mocap_data[(mocap_data.T != 0).any()]
        mocap_data = mocap_data.replace(0.0, np.NaN)
        mocap_data = mocap_data.interpolate(axis=0)
        mocap_data = mocap_data.bfill(axis=0)
        mocap_data = mocap_data.ffill(axis=0)

        # Downsampling to specific fps
        new_length = int(round((mocap_n_frame/250)*change_fps, 0))
        resampled = signal.resample(mocap_data, new_length)

        mocap_data_downsample = pd.DataFrame(
            resampled, columns=mocap_data.columns)
        print("mocap_data_downsample.shape:", mocap_data_downsample.shape)
        print("---")

        """
        #針對 joint 進行正規化
        normalization: 把每一個關節點設為原點: ex 左腳腳底(比較不會動 point 30 作為原點)，

        Naive:
        point 1 的 y 軸：假設人的身高是 1 (xxx, 1, zzz)
        """
        x_cols = [col for col in mocap_data_downsample.columns if '_X' in col]
        y_cols = [col for col in mocap_data_downsample.columns if '_Y' in col]
        z_cols = [col for col in mocap_data_downsample.columns if '_Z' in col]

        left_foot_test_x = mocap_data_downsample.loc[:, ["LANK_X"]]
        left_foot_test_y = mocap_data_downsample.loc[:, ["LANK_Y"]]
        left_foot_test_z = mocap_data_downsample.loc[:, ["LANK_Z"]]

        mocap_data_downsample.loc[:, x_cols] = mocap_data_downsample.loc[:, x_cols].sub(
            left_foot_test_x['LANK_X'], axis=0)
        mocap_data_downsample.loc[:, y_cols] = mocap_data_downsample.loc[:, y_cols].sub(
            left_foot_test_y['LANK_Y'], axis=0)
        mocap_data_downsample.loc[:, z_cols] = mocap_data_downsample.loc[:, z_cols].sub(
            left_foot_test_z['LANK_Z'], axis=0)

        head_x = mocap_data_downsample.loc[:, ["RFHD_X"]]
        head_y = mocap_data_downsample.loc[:, ["RFHD_Y"]]
        head_z = mocap_data_downsample.loc[:, ["RFHD_Z"]]

        people_height = head_z

        mocap_data_downsample.loc[:, x_cols] = mocap_data_downsample.loc[:, x_cols].div(
            people_height['RFHD_Z'], axis=0)
        mocap_data_downsample.loc[:, y_cols] = mocap_data_downsample.loc[:, y_cols].div(
            people_height['RFHD_Z'], axis=0)
        mocap_data_downsample.loc[:, z_cols] = mocap_data_downsample.loc[:, z_cols].div(
            people_height['RFHD_Z'], axis=0)

        ###
        Row_list = []

        # Iterate over each row to turn (34, 3) into (1, 102)
        for index, rows in mocap_data_downsample.iterrows():
            rows = rows.fillna(0)
            my_list = rows.values.tolist()
            my_list_per3 = [my_list[i:i+3] for i in range(0, len(my_list), 3)]
            Row_list.append(flatten(my_list_per3))

        print("Row_list len:", len(Row_list))
        keypoint_list.append(Row_list)
        keypoint_dictionary[this_performer][name_code] = Row_list
        
        target_dict[name_code] = np.asarray(Row_list)
        
        # motion_data_output = open('./preprocessed_data_save_new/motion/' +
        #                           name_code + '_motion_data.pkl', 'wb')
        # pickle.dump(np.asarray(Row_list), motion_data_output)
        # motion_data_output.close()


# Read Audio

mfcc_dictionary = {}
mfcc_length_list = []
annotation_extend_dict = {}

for this_performer in performaer_list:
    mfcc_dictionary[this_performer] = {}
    # UoE_violin_midi_path = "./UoE_violin_midi/violin_midi"
    UoE_violin_audio_path = "./UoE_violin_midi/performance_audio"
    audio_path_list = glob.glob(
        UoE_violin_audio_path + "/" + this_performer + "/*.wav")

    composer_code = {"Bach1_S1": "ba1",
                     "Bach2_S1": "ba3",
                     "Bach2_S2": "ba4",
                     "Beeth1_S1": "be4",
                     "Beeth2_S1": "be8",
                     "Elgar_S1": "el1",
                     "Flower_S1": "de1",
                     "Mend_S1": "me4",
                     "Mozart1_S1": "mo4",
                     "Mozart2_S1": "mo5",
                     "Wind_S1": "de2"}

    for item_path in audio_path_list:
        splited_name = str(item_path).split('/')
        name_code = splited_name[4].split('.')[0]
        # print("splited_name:", splited_name)
        print("name_code:", name_code)

        # read_piano_roll = read_midi(item_path, change_fps)
        read_audio_mfcc = audio_preprocess(item_path, change_fps)
        
        print("seq_len of audio data:", len(read_audio_mfcc[0]))
        mfcc_length_list.append(len(read_audio_mfcc[0]))
        read_mfcc_transpose = read_audio_mfcc

        motion_len = len(
            keypoint_dictionary[this_performer][name_code])
        audio_len = len(read_mfcc_transpose)
        if motion_len > audio_len:
            #(top,bottom), (left,right)
            read_mfcc_transpose = np.pad(read_mfcc_transpose,
                                               pad_width=((0, motion_len - audio_len), (0, 0)))
        if motion_len < audio_len:
            n = audio_len - motion_len
            read_mfcc_transpose = read_mfcc_transpose[:-n, :]

        ### read annotation
        performer_id, composer, section, try_count = name_code.split("_")

        try_count = try_count.replace("T", "")
        try_count = try_count.zfill(2)

        # Rename the file according to the desired format
        align_filename = f"{composer_code[str(composer + '_' + section)]}_{performer_id.replace('vio', 'ev')}_{try_count}_align_notetime.csv"
        cadence_filename = f"{composer_code[str(composer + '_' + section)]}_{performer_id.replace('vio', 'ev')}_{try_count}_cadence.csv"
        expression_filename = f"{composer_code[str(composer + '_' + section)]}_{performer_id.replace('vio', 'ev')}_{try_count}_expression.csv"
        phrase_filename = f"{composer_code[str(composer + '_' + section)]}_{performer_id.replace('vio', 'ev')}_{try_count}_phrase_section.csv"
        
        compoer_id_new = composer_code[str(composer + '_' + section)]
        performer_id_new = performer_id.replace('vio', 'ev')
        
        align_file_path = "./UoE_violin_midi/[Final_Ensure]ev/"+compoer_id_new +"/"+performer_id_new+"/"+try_count+"/annotations/"+align_filename
        cadence_file_path = "./UoE_violin_midi/[Final_Ensure]ev/"+compoer_id_new +"/"+performer_id_new+"/"+try_count+"/annotations/"+cadence_filename
        expression_file_path = "./UoE_violin_midi/[Final_Ensure]ev/"+compoer_id_new +"/"+performer_id_new+"/"+try_count+"/annotations/"+expression_filename
        phrase_file_path = "./UoE_violin_midi/[Final_Ensure]ev/"+compoer_id_new +"/"+performer_id_new+"/"+try_count+"/annotations/"+phrase_filename
        
        alignment_df = pd.read_csv(align_file_path, dtype={
            'onset': 'float64',
            'offset': 'float64',
            'score_position': 'float64',
            'score_position_offset': 'float64'
        })
        cadence_df = pd.read_csv(cadence_file_path, dtype={
            'onset': 'float64',
            'offset': 'float64'
        })
        expression_df = pd.read_csv(expression_file_path, dtype={
            'onset': 'float64',
            'offset': 'float64'
        })
        phrase_df = pd.read_csv(phrase_file_path, dtype={
            'onset': 'float64',
            'offset': 'float64'
        })

        audio_seq_len = len(read_mfcc_transpose.T[0])
        print("audio_seq_len", audio_seq_len)
        audio_time_df = pd.DataFrame(np.arange(0, (audio_seq_len * (1/change_fps)), (1/change_fps))[:, None])
        audio_time_df = audio_time_df.rename({0: 'time_stamp'}, axis=1)
        audio_time_df = audio_time_df.head(audio_seq_len)
        print("audio_time_df.shape:", audio_time_df.shape)

        # [1]
        # print(alignment_df[:5])
        # print(cadence_df[:5])
        # print(cadence_df)
        alignment_cadence_df = alignment_df.copy()
        alignment_cadence_df['cadence'] = 0#TODO need
        # cadence part mark as 1, else is 0 [0, 0, 0, 0, 0, 0, ..., 1, 1, 1, 1, 0, ...]
        for i, row in alignment_cadence_df.iterrows():
            # loop through each row in cadence_df
            for j, cadence_row in cadence_df.iterrows():
                # check if score_position is between onset and offset of cadence
                if row['score_position'] >= cadence_row['onset'] and row['score_position'] <= cadence_row['offset']:
                    # set cadence value to 1
                    alignment_cadence_df.at[i, 'cadence'] = 1
        # print("alignment_cadence_df\n",alignment_cadence_df)

        # [2]
        # Merge the phrase annotation dataframe with the alignment dataframe
        # Create a dictionary to map section values to section codes
        alignment_phrase_df = alignment_df.copy()
        alignment_phrase_df['phrase'] = 0#TODO need
        section_map = {}
        code = 1
        for section in phrase_df['Section'].unique():
            section_map[section] = code
            code += 1

        # Map the section values to section codes
        phrase_df['section_code'] = phrase_df['Section'].map(section_map)
        # print("new phrase_df\n", phrase_df)

        # Iterate through each row in the alignment data frame
        for i, row in alignment_phrase_df.iterrows():
            score_pos = row['score_position']
            # Iterate through each row in the phrase data frame
            for j, phrase_row in phrase_df.iterrows():
                onset = phrase_row['onset']
                offset = phrase_row['offset']
                section_code = phrase_row['section_code']
                # Check if the score position is between the phrase onset and offset
                if onset <= score_pos < offset:
                    # Set the phrase value for the alignment row
                    alignment_phrase_df.at[i, 'phrase'] = section_code

        # Print the updated alignment data frame
        # print("alignment_df\n", alignment_phrase_df)
        # the phrase mark with order [1, 1, 1, 1, ..., 2, 2, 2, ..., 3, 3, 3, 3, ...]
        # [3]
        alignment_expression_df = alignment_df.copy()
        alignment_expression_df['legato'] = 0#TODO need
        alignment_expression_df['staccato'] = 0#TODO need
        alignment_expression_df['accent'] = 0#TODO need
        alignment_expression_df['rit'] = 0#TODO need
        alignment_expression_df['accel'] = 0#TODO need
        alignment_expression_df['cresc'] = 0#TODO need
        alignment_expression_df['dim'] = 0#TODO need
        alignment_expression_df['dynamic'] = 0#TODO need
        
        dynamic_dict = {'ppp': 0.125, 'pp': 0.25, 'p': 0.375, 'mp': 0.5, 'mf': 0.625, 'f': 0.75, 'ff': 0.875, 'fff': 1}

        legato_rows = expression_df['expression'] == 'legato'
        for i, row in expression_df[legato_rows].iterrows():
            onset = row['onset']
            offset = row['offset']
            for j, expression_row in alignment_expression_df.iterrows():
                if expression_row['score_position'] >= row['onset'] and expression_row['score_position'] <= row['offset']:
                    alignment_expression_df.at[j, 'legato'] = 1
        
        legato_rows = expression_df['expression'] == 'staccato'
        for i, row in expression_df[legato_rows].iterrows():
            onset = row['onset']
            offset = row['offset']
            for j, expression_row in alignment_expression_df.iterrows():
                if expression_row['score_position'] >= row['onset'] and expression_row['score_position'] <= row['offset']:
                    alignment_expression_df.at[j, 'staccato'] = 1

        legato_rows = expression_df['expression'] == 'accent'
        for i, row in expression_df[legato_rows].iterrows():
            onset = row['onset']
            offset = row['offset']
            for j, expression_row in alignment_expression_df.iterrows():
                if expression_row['score_position'] >= row['onset'] and expression_row['score_position'] <= row['offset']:
                    alignment_expression_df.at[j, 'accent'] = 1

        legato_rows = expression_df['expression'] == 'rit'
        for i, row in expression_df[legato_rows].iterrows():
            onset = row['onset']
            offset = row['offset']
            for j, expression_row in alignment_expression_df.iterrows():
                if expression_row['score_position'] >= row['onset'] and expression_row['score_position'] <= row['offset']:
                    alignment_expression_df.at[j, 'rit'] = 1

        legato_rows = expression_df['expression'] == 'accel'
        for i, row in expression_df[legato_rows].iterrows():
            onset = row['onset']
            offset = row['offset']
            for j, expression_row in alignment_expression_df.iterrows():
                if expression_row['score_position'] >= row['onset'] and expression_row['score_position'] <= row['offset']:
                    alignment_expression_df.at[j, 'accel'] = 1

        legato_rows = expression_df['expression'] == 'cresc'
        for i, row in expression_df[legato_rows].iterrows():
            onset = row['onset']
            offset = row['offset']
            for j, expression_row in alignment_expression_df.iterrows():
                if expression_row['score_position'] >= row['onset'] and expression_row['score_position'] <= row['offset']:
                    alignment_expression_df.at[j, 'cresc'] = 1

        legato_rows = expression_df['expression'] == 'dim'
        for i, row in expression_df[legato_rows].iterrows():
            onset = row['onset']
            offset = row['offset']
            for j, expression_row in alignment_expression_df.iterrows():
                if expression_row['score_position'] >= row['onset'] and expression_row['score_position'] <= row['offset']:
                    alignment_expression_df.at[j, 'dim'] = 1

        for index, row in expression_df.iterrows():
            # Check if the expression is one of the dynamic values
            if row['expression'] in dynamic_dict:
                # Get the corresponding dynamic value
                dynamic_value = dynamic_dict[row['expression']]
                # Check if the score_position is within the onset and offset of the expression
                filter_expression = (alignment_expression_df['score_position'] >= row['onset']) & (alignment_expression_df['score_position'] < row['offset'])
                # Update the 'dynamic' column for the filtered rows
                alignment_expression_df.loc[filter_expression, 'dynamic'] = dynamic_value

        alignment_expression_df.loc[:,['dynamic']] = alignment_expression_df.loc[:,['dynamic']].replace(0.0, np.NaN)
        alignment_expression_df.loc[:,['dynamic']] = alignment_expression_df.loc[:,['dynamic']].ffill()
        alignment_expression_df.loc[:,['dynamic']] = alignment_expression_df.loc[:,['dynamic']].bfill()
        # print("alignment_expression_df\n", alignment_expression_df)
        
        audio_time_df['cadence'] = 0#alignment_cadence_df['cadence'].copy()
        audio_time_df['phrase'] = 0#alignment_phrase_df['phrase'].copy()
        audio_time_df['legato'] = 0#alignment_expression_df['legato'].copy()
        audio_time_df['staccato'] = 0#alignment_expression_df['staccato'].copy()
        audio_time_df['accent'] = 0#alignment_expression_df['accent'].copy()
        audio_time_df['rit'] = 0#alignment_expression_df['rit'].copy()
        audio_time_df['accel'] = 0#alignment_expression_df['accel'].copy()
        audio_time_df['cresc'] = 0#alignment_expression_df['cresc'].copy()
        audio_time_df['dim'] = 0#alignment_expression_df['dim'].copy()
        audio_time_df['dynamic'] = 0#alignment_expression_df['dynamic'].copy()
        
        # Loop through each row of the alignment_cadence_df dataframe
        for index, row in alignment_cadence_df.iterrows():
            # Find all rows in audio_time_df where the time_stamp value is between the onset and offset values
            audio_time_df.loc[(audio_time_df["time_stamp"] >= row['onset']) & (audio_time_df["time_stamp"] < row["offset"]), 'cadence'] = row["cadence"]

        for index, row in alignment_phrase_df.iterrows():
            # Find all rows in audio_time_df where the time_stamp value is between the onset and offset values
            audio_time_df.loc[(audio_time_df["time_stamp"] >= row['onset']) & (audio_time_df["time_stamp"] < row["offset"]), 'phrase'] = row["phrase"]
        
        for index, row in alignment_expression_df.loc[alignment_expression_df['legato'] == 1].iterrows():
            # Find all rows in audio_time_df where the time_stamp value is between the onset and offset values
            audio_time_df.loc[(audio_time_df["time_stamp"] >= row['onset']) & (audio_time_df["time_stamp"] < row["offset"]), 'legato'] = row['legato']
        
        for index, row in alignment_expression_df.loc[alignment_expression_df['staccato'] == 1].iterrows():
            # Find all rows in audio_time_df where the time_stamp value is between the onset and offset values
            audio_time_df.loc[(audio_time_df["time_stamp"] >= row['onset']) & (audio_time_df["time_stamp"] < row["offset"]), 'staccato'] = row['staccato']
        
        for index, row in alignment_expression_df.loc[alignment_expression_df['accent'] == 1].iterrows():
            # Find all rows in audio_time_df where the time_stamp value is between the onset and offset values
            audio_time_df.loc[(audio_time_df["time_stamp"] >= row['onset']) & (audio_time_df["time_stamp"] < row["offset"]), 'accent'] = row['accent']
        
        for index, row in alignment_expression_df.loc[alignment_expression_df['rit'] == 1].iterrows():
            # Find all rows in audio_time_df where the time_stamp value is between the onset and offset values
            audio_time_df.loc[(audio_time_df["time_stamp"] >= row['onset']) & (audio_time_df["time_stamp"] < row["offset"]), 'rit'] = row['rit']
        
        for index, row in alignment_expression_df.loc[alignment_expression_df['accel'] == 1].iterrows():
            # Find all rows in audio_time_df where the time_stamp value is between the onset and offset values
            audio_time_df.loc[(audio_time_df["time_stamp"] >= row['onset']) & (audio_time_df["time_stamp"] < row["offset"]), 'accel'] = row['accel']
        
        for index, row in alignment_expression_df.loc[alignment_expression_df['cresc'] == 1].iterrows():
            # Find all rows in audio_time_df where the time_stamp value is between the onset and offset values
            audio_time_df.loc[(audio_time_df["time_stamp"] >= row['onset']) & (audio_time_df["time_stamp"] < row["offset"]), 'cresc'] = row['cresc']
        
        for index, row in alignment_expression_df.loc[alignment_expression_df['dim'] == 1].iterrows():
            # Find all rows in audio_time_df where the time_stamp value is between the onset and offset values
            audio_time_df.loc[(audio_time_df["time_stamp"] >= row['onset']) & (audio_time_df["time_stamp"] < row["offset"]), 'dim'] = row['dim']
        
        for index, row in alignment_expression_df.iterrows():
            # Find all rows in audio_time_df where the time_stamp value is between the onset and offset values
            audio_time_df.loc[(audio_time_df["time_stamp"] >= row['onset']) & (audio_time_df["time_stamp"] < row["offset"]), 'dynamic'] = row['dynamic']
        ###
        
        # print(audio_time_df)
        # print(audio_time_df.describe())
        
        #different type of expression given different number:
        #legato = 1, when it occur then mark as 1, else 0
        #staccato = 2, when it occur then mark as 1, else 0
        #accent = 3, when it occur then mark as 1, else 0
        #rit = 4, when it occur then mark as 1, else 0
        #accel = 5, when it occur then mark as 1, else 0
        #cresc = 6, when it occur then mark as 1, else 0
        #dim = 7, when it occur then mark as 1, else 0
        #Dynamic (MIDI dynamics note velocity values)
        # ppp: 16
        # pp: 32
        # p: 48
        # mp: 64
        # mf: 80
        # f: 96
        # ff: 112
        # fff: 127

        #legato 圓滑線：圓滑線
        #staccato 斷奏：．, ▾
        #accent 重音：fz, sfz, rf, sf, >, –, ∧, rin(rinf)
        #rit.漸慢：rall, rallent, riten, lento, calando(漸弱、漸慢，目前 rit 和 dim 都標)
        #accel.漸快(新加入)：stringendo, accel.
        #cresc.漸強：cresc, ＜
        #dim.漸弱：dim, ＞, calando(漸弱、漸慢，目前 rit 和 dim 都標)
        #Dynamic 力度強弱：ppp, pp, p, mp, mf, f, ff, fff

        ### read annotation end
        print(audio_time_df['dynamic'].isnull().sum())

        # piano_roll_T = np.transpose(read_piano_roll, (0, 1))
        diff = np.diff(read_mfcc_transpose.T, axis=1)
        onset_info = np.sum((np.abs(diff) > 0).astype(int),  axis=0)
        onset_info_pad = np.pad(onset_info, (1, 0), 'constant')
        print("onset_info:", onset_info[None, :].shape)
        print("onset_info_pad:", onset_info_pad[None, :].shape)
        
        # input_data = np.concatenate((read_piano_roll, onset_info_pad), axis=1)
        # input_data = np.append(read_piano_roll_transpose.T, onset_info_pad[None, :], axis=0)
        
        annotation_extend = audio_time_df[['cadence', 'phrase', 'legato', 'staccato', 'accent', 'rit', 'accel', 'cresc', 'dim', 'dynamic']].to_numpy()
        print(annotation_extend.shape)
        
        annotation_extend_dict[name_code] = annotation_extend
        # input_data = np.append(read_piano_roll_transpose, annotation_extend.T, axis=0)
        # print("input_data", input_data.shape)
        ####
        print("read_mfcc_transpose.shape", read_mfcc_transpose.shape)
        mfcc_dictionary[this_performer][name_code] = read_mfcc_transpose
        # Save as pickle
        if not os.path.exists('./preprocessed_data_save_audio/audio/'):
            os.makedirs('./preprocessed_data_save_audio/audio/')
        
        audio_data_output = open('./preprocessed_data_save_audio/audio/' +
                                name_code + '_audio_data.pkl', 'wb')
        pickle.dump(read_mfcc_transpose, audio_data_output)
        audio_data_output.close()
        
        ####
        #append annotation to target data
        Row_list = target_dict[name_code]

        Row_list_extend = np.append(np.array(Row_list), annotation_extend_dict[name_code], axis=1)
        print("Row_list_extend.shape:", Row_list_extend.shape)
        keypoint_dictionary[this_performer][name_code] = Row_list_extend#np.array(Row_list)

        if not os.path.exists('./preprocessed_data_save_audio/motion/'):
            os.makedirs('./preprocessed_data_save_audio/motion/')
            
        motion_data_output = open('./preprocessed_data_save_audio/motion/' +
                                  name_code + '_motion_data.pkl', 'wb')
        pickle.dump(np.asarray(Row_list_extend), motion_data_output)
        motion_data_output.close()
        # load_pickle
        # audio_data_input = open('./preprocessed_data_save/audio/' +
        #                        this_performer + '_' + name_code + '_audio_data.pkl', 'rb')
        # audio_data = pickle.load(audio_data_input)
        # audio_data_input.close()


# mocap_data_downsample = pd.read_pickle('./preprocessed_data_save/motion_data.pkl')
# print("MIDI length:")
# print(piano_roll_length_list)
# print("Motion length:")
# for i in range(0, 110):
#     print(len(keypoint_list[i]))

each_audio_motion_len_compare = {}
for this_performer in performaer_list:
    for piece_name in piece_list:
        each_audio_motion_len_compare[this_performer + '_' + piece_name] = {}
        each_audio_motion_len_compare[this_performer + '_' + piece_name]["audio"] = len(
            mfcc_dictionary[this_performer][this_performer + '_' + piece_name])
        each_audio_motion_len_compare[this_performer + '_' + piece_name]["motion"] = len(
            keypoint_dictionary[this_performer][this_performer + '_' + piece_name])

print("each_audio_motion_len_compare")
print(each_audio_motion_len_compare)
#piano_roll_dictionary[this_performer][this_performer + '_' + piece_list[0]].shape
#keypoint_dictionary[this_performer][this_performer + '_' + piece_list[0]].shape
df_each_audio_motion_len_compare = pd.DataFrame(each_audio_motion_len_compare)
print(df_each_audio_motion_len_compare)
print(df_each_audio_motion_len_compare.T.to_markdown())
