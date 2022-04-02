from absl import app
from absl import flags
import json
from pyHGT.datax import HGTGraph
from utils.utils import get_files_into_dict
import networkx as nx

FLAGS = flags.FLAGS

flags.DEFINE_string('input_dir',
                    'dataset/output',
                    'dir of the graph input data')
flags.DEFINE_string('output_graph_file',
                    'output/graphs/OAG_graph.pk',
                    'dir of the graph oupt')
flags.DEFINE_string('graph_params_dir',
                    'config/HGT_graph_params_OAG.json',
                    'dir of the graph params')

def main(argv):
    # load graph parameters
    with open(FLAGS.graph_params_dir) as json_file:
        params = json.load(json_file)

    data = get_files_into_dict(FLAGS.input_dir, "\t")
    OAG_HGT_graph = HGTGraph(params, data)
    nx.write_gpickle(OAG_HGT_graph.G, FLAGS.output_graph_file)

if __name__ == '__main__':
    app.run(main)
