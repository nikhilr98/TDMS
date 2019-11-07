
# coding: utf-8

# # Raga ID using Time Delayed Melodic Surfaces

# In[110]:


import sys, csv
from essentia import *
from essentia.standard import *
from pylab import *
from numpy import *
import IPython
import numpy
from matplotlib.colors import Normalize
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import vamp
import sklearn.preprocessing


# In[111]:


audio = MonoLoader(filename='carnatic_varnam_1.0/Audio/223599__gopalkoduri__carnatic-varnam-by-dharini-in-saveri-raaga.mp3')()
audio = EqualLoudness()(audio)
sr=44100
IPython.display.Audio(data=audio, rate=44100)


# In[112]:


audio = MonoLoader(filename='carnatic_varnam_1.0/Audio/223582__gopalkoduri__carnatic-varnam-by-vignesh-in-abhogi-raaga.mp3')()
audio = EqualLoudness()(audio)
sr=44100
IPython.display.Audio(data=audio, rate=44100)


# In[113]:


#Loading the file.
def audio_load(name):
    audio = MonoLoader(filename=name)()
    audio = EqualLoudness()(audio)
    #audio_original=audio[start*44100:end*44100]
    sr=44100
    IPython.display.Audio(data=audio, rate=44100)
    return audio, name


# In[114]:


def feature_extraction(audio):
    
    #Melody estimation done using Melodia vamp plugin
    data = vamp.collect(audio, 44100, "mtg-melodia:melodia")
    hop, melody = data['vector']
    timestamps = 8 * 128/44100.0 + np.arange(len(melody)) * (128/44100.0)
    
    #Finding Tonic and Normalizing
    tonic=np.array(TonicIndianArtMusic()(audio))
    normalized_cent=1200*log2(melody/tonic)
    
    #Rejecting frames with invalid pitch values
    pitch_new=[]
    for i in range(0,len(normalized_cent)):
        if(not math.isnan(normalized_cent[i])):
            pitch_new.append(normalized_cent[i])
    pitch_np=numpy.array(pitch_new)
    
    #Using octave wrapping integer binnning function
    #B=ceil(floor(pitch_np)/10)%120
    B=floor((pitch_np/10)%120)
    
    return B


# In[115]:


def surface(B,delay):
    convertor=np.tile(np.arange(0,120),(len(B[:-delay]),1))
    I_original=np.equal(np.tile(B[:-delay], (120,1)), convertor.T)
    I_delayed=np.equal(np.tile(B[delay:], (120,1)), convertor.T)
    I_original=I_original.astype(int)
    I_delayed=I_delayed.astype(int)
    S=np.matmul(I_original,I_delayed.T)
    return S


# In[116]:


def plot_figure(S, name):
    get_ipython().magic('matplotlib notebook')
    fig = plt.figure()
    fig.suptitle(t=name, fontsize=16)
    ax = fig.gca(projection='3d')
    X = np.arange(0,120)
    Y = np.arange(0,120)
    X, Y = np.meshgrid(X, Y)
    surf = ax.plot_surface(X, Y, S, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()


# # Surface plot for given Raga

# The peaks along the diagonal of this plot give the corresponding bins of the swaras that the raga is comprised of. The dynamic range is quite high since Gaussian smoothening hasn't been applied yet.  

# In[168]:


audio1, name1=audio_load(name='carnatic_varnam_1.0/Audio/223604__gopalkoduri__carnatic-varnam-by-ramakrishnamurthy-in-sri-raaga.mp3')
audio2, name2=audio_load(name='carnatic_varnam_1.0/Audio/223603__gopalkoduri__carnatic-varnam-by-badrinarayanan-in-sri-raaga.mp3')
audio3, name3=audio_load(name='carnatic_varnam_1.0/Audio/223605__gopalkoduri__carnatic-varnam-by-vignesh-in-sri-raaga.mp3')
audio4, name4=audio_load(name='carnatic_varnam_1.0/Audio/223601__gopalkoduri__carnatic-varnam-by-ramakrishnamurthy-in-saveri-raaga.mp3')
audio5, name5=audio_load(name='carnatic_varnam_1.0/Audio/223602__gopalkoduri__carnatic-varnam-by-sreevidya-in-saveri-raaga.mp3')


# In[169]:


B1=feature_extraction(audio1)
B2=feature_extraction(audio2)
B3=feature_extraction(audio3)
#B4=feature_extraction(audio4)
#B5=feature_extraction(audio5)


# In[170]:


S1=surface(B1,50)
S2=surface(B2,50)
S3=surface(B3,50)
#S4=surface(B4,50)
#S5=surface(B5,50)


# In[174]:


plot_figure(S1, name1) 


# In[175]:


plot_figure(S2, name2)



# In[176]:


plot_figure(S3, name3)


# In[166]:


plot_figure(S4, name4)


# In[167]:


plot_figure(S5,name5)


# In[125]:


numpy.set_printoptions(threshold=sys.maxsize)


# In[126]:


print(B5)


# In[127]:


fig=plt.figure()

plt.hist(B5,120)
plt.show()

