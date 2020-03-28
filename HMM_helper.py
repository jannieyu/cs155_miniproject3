########################################
# CS/CNS/EE 155 2018
# Problem Set 6
#
# Author:       Andrew Kang
# Description:  Set 6 HMM helper
########################################

import re
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from matplotlib import animation
from matplotlib.animation import FuncAnimation
import os



####################
# WORDCLOUD FUNCTIONS
####################

def mask():
    # Parameters.
    r = 128
    d = 2 * r + 1

    # Get points in a circle.
    y, x = np.ogrid[-r:d-r, -r:d-r]
    circle = (x**2 + y**2 <= r**2)

    # Create mask.
    mask = 255 * np.ones((d, d), dtype=np.uint8)
    mask[circle] = 0

    return mask

def text_to_wordcloud(text, max_words=50, title='', show=True):
    plt.close('all')

    # Generate a wordcloud image.
    wordcloud = WordCloud(random_state=0,
                          max_words=max_words,
                          background_color='white',
                          mask=mask()).generate(text)

    # Show the image.
    if show:
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=24)
        plt.show()

    return wordcloud

def states_to_wordclouds(hmm, obs_map, max_words=50, show=True):
    # Initialize.
    M = 100000
    n_states = len(hmm.A)
    obs_map_r = obs_map_reverser(obs_map)
    syllable_dictionary = get_syllable_dict()
    wordclouds = []

    # Generate a large emission.
    emission, states = hmm.generate_emission(M, obs_map_r, syllable_dictionary)

    # For each state, get a list of observations that have been emitted
    # from that state.
    obs_count = []
    for i in range(n_states):
        obs_lst = np.array(emission)[np.where(np.array(states) == i)[0]]
        obs_count.append(obs_lst)

    # For each state, convert it into a wordcloud.
    for i in range(n_states):
        obs_lst = obs_count[i]
        sentence = [obs_map_r[j] for j in obs_lst]
        sentence_str = ' '.join(sentence)

        wordclouds.append(text_to_wordcloud(sentence_str, max_words=max_words, title='State %d' % i, show=show))

    return wordclouds


####################
# HMM FUNCTIONS
####################


def parse_observations(text):

    # Create dictionary of all possible words in shakespeare
    file = open(os.path.join(os.getcwd(), 'data/Syllable_dictionary.txt')).read()
    punctation_lines = [line.split() for line in file.split('\n') if line.split()]
    stripped_dict = {}
    for line in punctation_lines:
        stripped_dict[re.sub(r'[^\w]', '', line[0])] = line[0]

    # Create dictionary of all words with unique id
    lines = [line.split() for line in text.split('\n') if line.split()]
    obs_counter = 6
    obs = []
    
    obs_map = {".": 0, "?": 1, ",": 2, ":": 3, "!": 4, " ": 5}
  

    for line in lines:
        obs_elem = []

        if len(line) != 1:
            
            
            for word in line:
                word = stripped_dict[re.sub(r'[^\w]', '', word.lower())]
                
                if word not in obs_map:
                    obs_map[word] = obs_counter
                    obs_counter += 1
                
                obs_elem.append(obs_map[word])


            lastword = line[len(line) - 1]
            if lastword[len(lastword) - 1] in [".", "?", ",", ":", "!"]:
                obs_elem.append(obs_map[lastword[len(lastword) - 1]])
            else:
                obs_elem.append(5)



            obs.append(obs_elem)

  
    return obs, obs_map


def obs_map_reverser(obs_map):
    obs_map_r = {}

    for key in obs_map:
        obs_map_r[obs_map[key]] = key

    return obs_map_r

def sample_sentence(hmm, obs_map, n_syllables=100, first=None, reverse=False):

    # Get reverse map.
    obs_map_r = obs_map_reverser(obs_map)
    syllable_dictionary = get_syllable_dict()
    
    # Sample and convert sentence.
    emission, states = hmm.generate_emission(n_syllables, obs_map_r, syllable_dictionary, first)
    sentence = [obs_map_r[i] for i in emission]
    
    if reverse:
      punctuation = sentence[0]
      sentence.remove(punctuation)
      sentence.reverse()
    else:
      punctuation = sentence[len(sentence) - 1]
      sentence.remove(punctuation)


    # Capitalize all occurrences of 'i' and 'i'll'
    for i in range(len(sentence)):
      if sentence[i] in ['i', "i'll"]:
        sentence[i] = sentence[i].capitalize()

    # Join the words in the sentence by spaces
    sentence = ' '.join(sentence).strip()



    # Capitalize the first letter and return
    return sentence[0].capitalize() + sentence[1:] + str(punctuation)

def get_syllable_dict():

    file = open(os.path.join(os.getcwd(), 'data/Syllable_dictionary.txt')).read()
    lines = [line.split() for line in file.split('\n') if line.split()]
    syllables= {}

    #go through each line
    for line in lines:

        num_syllables = []
        end = []

        # go through each thing in word
        for i in range(1, len(line)):
            if line[i][0] == 'E':
                end.append(int(line[i][1]))
            else:
                num_syllables.append(int(line[i][0]))
        syllables[line[0]] = [num_syllables, end]

    return syllables

def get_rhyme_dict(text):
  # Split text into poems
  poems = re.split('\d+\s' , text)
  rhyme_dict = {}

  # List of line numbers which rhyme with each other
  rhyming_indices = [(0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 13)]

  for poem in poems:
    # Split poem into lines
    lines = [re.sub(r'[^\w\s]', '', line.strip().lower()) for line in poem.strip().split("\n")]

    # Ignore non-standard sonnets which don't have exactly 14 lines
    if len(lines) != 14:
      continue

    # Add last words of each pair of rhyming lines to rhyming dictionary
    for pair in rhyming_indices:
      # Get last words of both lines
      word1, word2 = lines[pair[0]].split(' ')[-1], lines[pair[1]].split(' ')[-1]

      # Add word2 to rhyming list of word1
      if word1 not in rhyme_dict:
        rhyme_dict[word1] = []
      rhyme_dict[word1].append(word2)

      # Add word1 to rhyming list of word2
      if word2 not in rhyme_dict:
        rhyme_dict[word2] = []
      rhyme_dict[word2].append(word1)

  return rhyme_dict

def get_rhyming_sentences(hmm, obs_map, rhyme_dict):
  # Choose two rhyming words randomly
  w1 = np.random.choice(list(rhyme_dict.keys()))
  w2 = np.random.choice(rhyme_dict[w1])

  # Generate sample sentences which end in the rhyming words
  s1 = sample_sentence(hmm, obs_map, n_syllables=10, first=obs_map[w1], reverse=True)
  s2 = sample_sentence(hmm, obs_map, n_syllables=10, first=obs_map[w2], reverse=True)

  return s1, s2

def sample_quatrain(hmm, obs_map, rhyme_dict):
  # Generate two pairs of rhyming sentences
  rhyme1 = get_rhyming_sentences(hmm, obs_map, rhyme_dict)
  rhyme2 = get_rhyming_sentences(hmm, obs_map, rhyme_dict)

  # Alternate rhymes for ABAB rhyme scheme
  return '\n'.join([rhyme1[0], rhyme2[0], rhyme1[1], rhyme2[1]])

def sample_rhyming_sonnet(hmm, obs_map, rhyme_dict):
  # Combine three quatrains and a couplet to make a sonnet
  quatrain1 = sample_quatrain(hmm, obs_map, rhyme_dict)
  quatrain2 = sample_quatrain(hmm, obs_map, rhyme_dict)
  quatrain3 = sample_quatrain(hmm, obs_map, rhyme_dict)
  couplet = '\n'.join(get_rhyming_sentences(hmm, obs_map, rhyme_dict))
  
  return '\n'.join([quatrain1, quatrain2, quatrain3, couplet])

####################
# HMM VISUALIZATION FUNCTIONS
####################

def visualize_sparsities(hmm, O_max_cols=50, O_vmax=0.1):
    plt.close('all')
    plt.set_cmap('viridis')

    # Visualize sparsity of A.
    plt.imshow(hmm.A, vmax=1.0)
    plt.colorbar()
    plt.title('Sparsity of A matrix')
    plt.show()

    # Visualize parsity of O.
    plt.imshow(np.array(hmm.O)[:, :O_max_cols], vmax=O_vmax, aspect='auto')
    plt.colorbar()
    plt.title('Sparsity of O matrix')
    plt.show()


####################
# HMM ANIMATION FUNCTIONS
####################

def animate_emission(hmm, obs_map, M=8, height=12, width=12, delay=1):
    # Parameters.
    lim = 1200
    text_x_offset = 40
    text_y_offset = 80
    x_offset = 580
    y_offset = 520
    R = 420
    r = 100
    arrow_size = 20
    arrow_p1 = 0.03
    arrow_p2 = 0.02
    arrow_p3 = 0.06
    
    # Initialize.
    n_states = len(hmm.A)
    obs_map_r = obs_map_reverser(obs_map)
    wordclouds = states_to_wordclouds(hmm, obs_map, max_words=20, show=False)

    # Initialize plot.    
    fig, ax = plt.subplots()
    fig.set_figheight(height)
    fig.set_figwidth(width)
    ax.grid('off')
    plt.axis('off')
    ax.set_xlim([0, lim])
    ax.set_ylim([0, lim])

    # Plot each wordcloud.
    for i, wordcloud in enumerate(wordclouds):
        x = x_offset + int(R * np.cos(np.pi * 2 * i / n_states))
        y = y_offset + int(R * np.sin(np.pi * 2 * i / n_states))
        ax.imshow(wordcloud.to_array(), extent=(x - r, x + r, y - r, y + r), aspect='auto', zorder=-1)

    # Initialize text.
    text = ax.text(text_x_offset, lim - text_y_offset, '', fontsize=24)
        
    # Make the arrows.
    zorder_mult = n_states ** 2 * 100
    arrows = []
    for i in range(n_states):
        row = []
        for j in range(n_states):
            # Arrow coordinates.
            x_i = x_offset + R * np.cos(np.pi * 2 * i / n_states)
            y_i = y_offset + R * np.sin(np.pi * 2 * i / n_states)
            x_j = x_offset + R * np.cos(np.pi * 2 * j / n_states)
            y_j = y_offset + R * np.sin(np.pi * 2 * j / n_states)
            
            dx = x_j - x_i
            dy = y_j - y_i
            d = np.sqrt(dx**2 + dy**2)

            if i != j:
                arrow = ax.arrow(x_i + (r/d + arrow_p1) * dx + arrow_p2 * dy,
                                 y_i + (r/d + arrow_p1) * dy + arrow_p2 * dx,
                                 (1 - 2 * r/d - arrow_p3) * dx,
                                 (1 - 2 * r/d - arrow_p3) * dy,
                                 color=(1 - hmm.A[i][j], ) * 3,
                                 head_width=arrow_size, head_length=arrow_size,
                                 zorder=int(hmm.A[i][j] * zorder_mult))
            else:
                arrow = ax.arrow(x_i, y_i, 0, 0,
                                 color=(1 - hmm.A[i][j], ) * 3,
                                 head_width=arrow_size, head_length=arrow_size,
                                 zorder=int(hmm.A[i][j] * zorder_mult))

            row.append(arrow)
        arrows.append(row)

    emission, states = hmm.generate_emission(M)

    def animate(i):
        if i >= delay:
            i -= delay

            if i == 0:
                arrows[states[0]][states[0]].set_color('red')
            elif i == 1:
                arrows[states[0]][states[0]].set_color((1 - hmm.A[states[0]][states[0]], ) * 3)
                arrows[states[i - 1]][states[i]].set_color('red')
            else:
                arrows[states[i - 2]][states[i - 1]].set_color((1 - hmm.A[states[i - 2]][states[i - 1]], ) * 3)
                arrows[states[i - 1]][states[i]].set_color('red')

            # Set text.
            text.set_text(' '.join([obs_map_r[e] for e in emission][:i+1]).capitalize())

            return arrows + [text]

    # Animate!
    print('\nAnimating...')
    anim = FuncAnimation(fig, animate, frames=M+delay, interval=1000)

    return anim

    # honestly this function is so jank but who even fuckin cares
    # i don't even remember how or why i wrote this mess
    # no one's gonna read this
    # hey if you see this tho hmu on fb let's be friends