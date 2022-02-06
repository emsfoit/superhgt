from absl import app
from absl import flags
import json
from pyHGT.data import HGTGraph
from utils.utils import get_files_into_dict
import dill

FLAGS = flags.FLAGS

flags.DEFINE_string('input_dir',
                    'dataset/output',
                    'dir of the graph input data')
flags.DEFINE_string('output_graph_file',
                    'graphs/OAG_graph.pk',
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

    dill.dump(OAG_HGT_graph.graph, open(FLAGS.output_graph_file, 'wb'))


if __name__ == '__main__':
    app.run(main)
