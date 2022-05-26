# Copyright 2015 Abhinav Maurya

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

 
import fileinput
#from sets import Set
import random
import time

import scipy.special
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy.stats import beta
import pprint, pickle
from datetime import datetime


resultspath = '../results_tot/'


def VisualizeTopics(phi, words, num_topics, viz_threshold=9e-3):
	phi_viz = np.transpose(phi)
	words_to_display = ~np.all(phi_viz <= viz_threshold, axis=1)
	words_viz = [words[i] for i in range(len(words_to_display)) if words_to_display[i]]
	phi_viz = phi_viz[words_to_display]

	fig, ax = plt.subplots()
	heatmap = plt.pcolor(phi_viz, cmap=plt.cm.Blues, alpha=0.8)
	plt.colorbar()
	plt.subplots_adjust(left=0.20)

	#fig.set_size_inches(8, 11)
	ax.grid(False)
	ax.set_frame_on(False)

	ax.set_xticks(np.arange(phi_viz.shape[1]) + 0.5, minor=False)
	ax.set_yticks(np.arange(phi_viz.shape[0]) + 0.5, minor=False)
	ax.invert_yaxis()
	ax.xaxis.tick_top()
	#plt.xticks(rotation=45)
	
	for t in ax.xaxis.get_major_ticks():
	    t.tick1On = False
	    t.tick2On = False
	for t in ax.yaxis.get_major_ticks():
	    t.tick1On = False
	    t.tick2On = False

	column_labels = words_viz	#['Word ' + str(i) for i in range(1,1000)]
	# row_labels = ['Topic ' + str(i+1) for i in range(0, num_topics)]
	row_labels = ['Coronavírus', 'Covid19 (Resíduos)', 'Infos Pandemia', 'Medidas de Proteção', 'Combate à Pandemia']
	ax.set_xticklabels(row_labels, minor=False)
	plt.xticks(fontsize=9, rotation=13)
	ax.set_yticklabels(column_labels, minor=False)

	plt.savefig(resultspath + 'topic_distribution3.png')
	plt.show()

def VisualizeEvolution(psi, timestamp):
	ts_list = timestamp.unique()
	xs = np.linspace(0, 1, num=ts_list.size)
	xs2 = [datetime.fromtimestamp(x).date() for x in ts_list]

	fig, ax = plt.subplots()
	row_labels = ['Coronavírus', 'Covid19 (Resíduos)', 'Infos Pandemia', 'Medidas de Proteção', 'Combate à Pandemia']

	for i in range(len(psi)):
		ys = [math.pow(1-x, psi[i][0]-1) * math.pow(x, psi[i][1]-1) / scipy.special.beta(psi[i][0],psi[i][1]) for x in xs]
		# ax.plot(xs2, ys, label='Topic ' + str(i+1))
		ax.plot(xs2, ys, label=row_labels[i])

	ax.legend(loc='best', frameon=False)
	plt.xticks(rotation=45)
	plt.subplots_adjust(bottom=0.15)
	plt.title("Probabilidade de cada tópico ao longo do tempo")

	plt.savefig(resultspath + 'topic_evolution3.png')
	plt.show()

def main():
	start_time = time.time()

	resultspath = '../results_tot/'
	tot_pickle_path = resultspath + 'covid_totnew.pickle'

	tot_pickle = open(tot_pickle_path, 'rb')
	par = pickle.load(tot_pickle)
	VisualizeTopics(par['n'], par['word_token'], par['T'])
	VisualizeEvolution(par['psi'], par['original_ts'])


	print("\nExecution time: {} seconds or {} minutes".format((time.time() - start_time), (time.time() - start_time)/60))

if __name__ == "__main__":
    main()
