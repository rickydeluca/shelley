import numpy as np

from shelley.utils.sm_solvers.Extended_Stable_Marriage_Algorithm import \
    ExtendedStableMarriage


class SMDataProcessor:
    def __init__(self,sim_matrix_src, sim_matrix_tgt):
        self.src_is_male = False
        self.male_choices = sim_matrix_src
        self.female_choices = sim_matrix_tgt
        self.female_length=len(sim_matrix_tgt)
        if (len(sim_matrix_src) <= len(sim_matrix_tgt)):
            self.male_choices = sim_matrix_src
            self.female_choices = sim_matrix_tgt
            self.src_is_male = True
        else:
            self.male_choices = sim_matrix_tgt
            self.female_choices = sim_matrix_src
            self.src_is_male = False
        processed_preferences = self.process_sm_data(self.male_choices, self.female_choices, len(self.male_choices), len(self.female_choices))
        self.male_preferences =processed_preferences['male_preferences']
        self.female_preferences = processed_preferences['female_preferences']
        
        esm = ExtendedStableMarriage(self.male_preferences,self.female_preferences, len(self.male_choices),
                                     len(self.female_choices))
        self.marriage = esm.MOSM(len(self.male_choices), len(self.female_choices))
        self.male_counter=esm.Get_Male_Counter()
        self.row, self.col = self.create_stable_marriage_permutation_matrix(self.marriage, len(self.male_choices), len(self.female_choices))

    def process_similarity_matrices(self,sim_matrix_src, sim_matrix_tgt):
        src_is_male = False
        if (len(sim_matrix_src) <= len(sim_matrix_tgt)):
            male_choices = sim_matrix_src
            female_choices = sim_matrix_tgt
            src_is_male = True
        else:
            male_choices = sim_matrix_tgt
            female_choices = sim_matrix_src

        return {'male_choices': male_choices, 'female_choices': female_choices, 'src_is_male': src_is_male}

    def process_sm_data(self,male_choices, female_choices, n, k):
        male_preferences = []
        female_preferences = []
        dummy_preferences = [k + 1] * k
        male_preferences.append(dummy_preferences)
        for i in range(n):
            preferences = np.argsort(male_choices[i])[::-1]
            male_preferences.append(preferences)
            male_preferences_np = np.array(male_preferences,dtype=object)
       
        for i in range(k):
            preferences = np.argsort(female_choices[i])[::-1]
            female_preferences.append(preferences)
            female_preferences_np = np.array(female_preferences) + 1  # adding 1 considering the
            
        output = []
        output.append(male_preferences_np)
        output.append(female_preferences_np)
        return {'male_preferences': male_preferences_np, 'female_preferences': female_preferences_np}

    def create_stable_marriage_permutation_matrix(self,female_marriage, n, k):
        s = np.arange(n * k)
        s = s.reshape((n, k))
        row = []
        col = []
        for i in range(len(female_marriage)):
            marriage_male_index=female_marriage[i]
            if(marriage_male_index !=0): #ignore the dummy male
                row.append(female_marriage[i] - 1)  # handling the dummy male (-1)
                col.append(i)
        
        return row,col

    def get_row_col_of_stable_marriage_permutation(self):
        
        if self.src_is_male:
            return self.row,self.col
        else:
            return self.col,self.row

    def get_stable_marriage_attention(self):
        male_counter_final=self.male_counter[1:] #ignore the dummy male
        male_preferences_final = self.male_preferences[1:] #ignore the dummy male
        male_counter_length=len(male_counter_final)
        male_preferences_length = len(male_preferences_final)
        attention_matrix=np.zeros((len(self.male_choices),len(self.female_choices)))
        for i in range(male_preferences_length):
            count=male_counter_final[i]
            
            if((count-self.female_length)>0):
                count=self.female_length
            for j in range(count):
                attention_matrix[i,male_preferences_final[i][j]]=1
        if self.src_is_male:
            return attention_matrix
        else:
            return attention_matrix.transpose() #return transpose if male src is false

