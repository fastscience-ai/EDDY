from create_explanations import load_model
import streamlit as st
import pickle
from torch_geometric.utils.loop import add_remaining_self_loops
from molecular_graph import Molecule
import matplotlib.pyplot as plt
from vis_utils import relevance_vis_2d_v1
from create_explanations import from_smiles
from data_from_smile import process_out
import math

from rdkit import Chem, rdBase
from rdkit.Chem import rdDepictor
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

import streamlit as st
import pickle
from torch_geometric.utils.loop import add_remaining_self_loops

from molecular_graph import Molecule
from dig.xgraph.evaluation import XCollector, ExplanationProcessor


def load(data):
    if data == "tox":
        with open('explanations_tox.pkl', 'rb') as f:
            loaded_dict = pickle.load(f)
        return loaded_dict
    else:
        with open('explanations_clintox.pkl', 'rb') as f:
            loaded_dict = pickle.load(f)
        return loaded_dict

tox_21_explanations = load(data="tox")
clintox_explanations = load(data="clintox")
#print(len(list(clintox_explanations.keys())))
def create_relevance_from_data(data, method="deeplift"):
    #print("Inside relevence from data Function")
    #print(method, data)
    relevances_scale = []
    all_edges = add_remaining_self_loops(data.edge_index)[0]
    if method == "deeplift":    
        try:
            exs = tox_21_explanations[data.smiles[0]]['deep_lift_edge_level_explanations']
            for id in range(all_edges.size()[1]):
                relevances_scale.append(([all_edges[0][id].item(),all_edges[1][id].item()], exs[1][id].item()-exs[0][id].item()) )
        except Exception as e:
            return []
        
    elif method == "gradcam":
        try:
            exs = tox_21_explanations[data.smiles[0]]['gradCam_edge_level_explanations']
            for id in range(all_edges.size()[1]):
                relevances_scale.append(([all_edges[0][id].item(),all_edges[1][id].item()], exs[1][id].item()-exs[0][id].item()) )
        except Exception as e:
            return []
            
    elif method=="gnnexplainer":
        try:
            exs = tox_21_explanations[data.smiles[0]]['gnnExplainer_edge_level_explanations']
            for id in range(all_edges.size()[1]):
                relevances_scale.append(([all_edges[0][id].item(),all_edges[1][id].item()], exs[1][id].item()-exs[0][id].item()) )
        except Exception as e:
            return []

    elif method == "gnnlrp":
        try:
            exs = tox_21_explanations[data.smiles[0]]['gnn_lrp_path_level_explanations']
            #print(exs)
            for id in range(exs['ids'].size()[0]): #scores
                node_list = []
                
                for idx, edge in enumerate(exs['ids'][id].tolist()):
                    if idx==0:
                        node_list.append(all_edges[0][edge].item())
                        node_list.append(all_edges[1][edge].item())
                    else:
                        node_list.append(all_edges[1][edge].item())
                relevances_scale.append((node_list, exs['score'][id][1].item()-exs['score'][id][0].item()) )
        except Exception as e:
            return []
    return relevances_scale

def create_relevance_from_explanation(data, exs, method="deeplift"):
    #print(method, data)
    relevances_scale = []
    all_edges = add_remaining_self_loops(data.edge_index)[0]
    if method == "deeplift":    
        try:
            for id in range(all_edges.size()[1]):
                relevances_scale.append(([all_edges[0][id].item(),all_edges[1][id].item()], exs[1][id].item()-exs[0][id].item()) )
        except Exception as e:
            return []
        
    elif method == "gradcam":
        try:
            for id in range(all_edges.size()[1]):
                relevances_scale.append(([all_edges[0][id].item(),all_edges[1][id].item()], exs[1][id].item()-exs[0][id].item()) )
        except Exception as e:
            return []
            
    elif method=="gnnexplainer":
        try:
            for id in range(all_edges.size()[1]):
                relevances_scale.append(([all_edges[0][id].item(),all_edges[1][id].item()], exs[1][id].item()-exs[0][id].item()) )
        except Exception as e:
            return []

    elif method == "gnnlrp":
        try:
            #print("comes to gnnlrp")
            #print(exs)
            for id in range(exs['ids'].size()[0]): #scores
                node_list = []
                
                for idx, edge in enumerate(exs['ids'][id].tolist()):
                    if idx==0:
                        node_list.append(all_edges[0][edge].item())
                        node_list.append(all_edges[1][edge].item())
                    else:
                        node_list.append(all_edges[1][edge].item())
                relevances_scale.append((node_list, exs['score'][id][1].item()-exs['score'][id][0].item()) )
        except Exception as e:
            return []
    return relevances_scale


neg_color = (0.0, 1.0, 0.0)
pos_color = (1.0, 0.0, 0.0)


def plot_rdkit(smi, relevances_scale, data, label='', fmt='svg'):
    rdDepictor.SetPreferCoordGen(True)
    mol = Chem.MolFromSmiles(smi)
    mol = Draw.PrepareMolForDrawing(mol)

    alist = []
    blist = []
    q = Chem.MolFromSmarts(smi)
    alist = [i for i in range(mol.GetNumAtoms())]
    #for match in mol.GetSubstructMatches(q):
    #    alist.extend(match)
    
    #print(data)
    all_edges = add_remaining_self_loops(data.edge_index)[0]
    edge_rel_map = {}
    for id in range(all_edges.size()[1]):
            edge_rel_map[(all_edges[0][id].item(),all_edges[1][id].item())]= relevances_scale[id]
    #print(edge_rel_map)
    print(alist)
    bcols = {}
    for ha1 in alist:
        for ha2 in alist:
            #print("comes, ", ha1, ha2)
            if ha1 > ha2:
                b = mol.GetBondBetweenAtoms(ha1, ha2)
                #print(ha1, ha2)
                if b:
                    #blist.append(b.GetIdx())
                    if ha2 not in bcols.keys():
                        bcols[ha2] = edge_rel_map[(ha1, ha2)][1]
                    else:
                        bcols[ha2]+= edge_rel_map[(ha1, ha2)][1]
                    
                    #if ha1 not in bcols.keys():
                    #    bcols[ha1] = edge_rel_map[(ha2, ha1)][1]
                    #else:
                    #    bcols[ha1]+= edge_rel_map[(ha2, ha1)][1]
    
    keys = bcols.keys()
    for key in keys:
        if bcols[key]>0:
            bcols[key] = [pos_color]
        else:
            bcols[key] = [neg_color]

    print(bcols)      
    acols = {}
    
    h_rads = {}
    h_lw_mult = {}

    for idx , (walk, rel) in enumerate(relevances_scale):
        for el in walk:
            if el not in acols.keys():
                acols[el] = 0.0
            else:
                acols[el] += rel
    keys = list(acols.keys())    
    for key in keys:
        if acols[key]<0:
            new_color = neg_color[:]
            #for i in range(len(new_color)):
            #    new_color[i]*=acols[key]
            acols[key] = [neg_color]
        else:
            acols[key] = [pos_color]
    if fmt == 'svg':
        d = rdMolDraw2D.MolDraw2DSVG(500, 500)
        mode = 'w'
    elif fmt == 'png':
        d = rdMolDraw2D.MolDraw2DCairo(500, 500)
        mode = 'wb'
    else:
        print('unknown format {}'.format(fmt))
        return
    
    d.drawOptions().fillHighlights = True
    d.DrawMoleculeWithHighlights(mol, label, acols, bcols, h_rads, h_lw_mult, -1)
    d.FinishDrawing()
    #print(d)
    with open("rdkit_temp_image.svg", mode) as f:
        #d.Get
        f.write(d.GetDrawingText())


def plot_streamlit(option,relevances_scale, actual = None, x_collector=None, data_source=None, model_type=None, draw_rdkit=True, data=None):
    if Vis_Style=="Custom":
        draw_rdkit=False
    #plt.show()
    if draw_rdkit:
        plot_rdkit(option,relevances_scale,data)
        st.image("rdkit_temp_image.svg")
    else:
        molecule = Molecule(option)
        sample, pos_2d, graph = molecule.load_molecule()
        #print(pos_2d)
        fig = plt.figure(figsize=(14, 8))
        ax = plt.subplot(1, 1, 1)
        #print(sample["_atomic_numbers"][0])
        ax = relevance_vis_2d_v1(ax, relevances_scale, sample["_atomic_numbers"][0], pos_2d, graph, shrinking_factor=10)
        plt.axis('off')
        st.pyplot(fig=plt)
    
    st.markdown('<p style="font-family:sans-serif; color:#66ff66; font-size: 20px;">Negative Contribution</p> ', unsafe_allow_html=True)
    st.markdown('<p style="font-family:sans-serif; color:#ff66ff; font-size: 20px;">Positive Contribution</p>', unsafe_allow_html=True)
    if x_collector:
        st.write("Fidelity is :", round(x_collector.fidelity, 2))
        st.write("Fidelity_inv is :", round(x_collector.fidelity_inv, 2))
        st.write("Sparsity is :", round(x_collector.sparsity, 2))
    
    #print(option)
    data = process_out(option)[0]
    model = load_model(data=data_source, model_type=model_type)
    out = model(data.x, data.edge_index)

    prediction = ""
    if out[0][0].item()>out[0][1].item():
        if data_source=="tox":
            prediction = "Non Toxic"
        else:
            prediction = "FDA Approved"
    else:
        if data_source=="tox":
            prediction = "Toxic"
        else:
            prediction = "FDA Rejected"
    print(data_source, prediction)
    print(out[0][0].item(),out[0][1].item(), actual)
    
    st.write("Prediction is :", prediction)
    if actual:
        st.write("Truth is :", actual)




#st.set_page_config(
#    page_title="Visual Framework to Benchmark Explanation Methods for Graph Neural Network based Drug Screening Models ",
#    page_icon="🎈",
#)
st.title("🔑 Visual Framework to Benchmark Explanation Methods for Graph Neural Network based Drug Screening Models ")

st.sidebar.title("Options")

DataSet = st.sidebar.radio(
        "Choose your Dataset",
        ["Tox21", "Clintox"],
        help="Tox21 predicts Nr-AhR probability and Clintox predicts the probability of FDA Approval",horizontal=True
)
ExplainerType = st.sidebar.radio(
        "Choose your explainer",
        ["DeepLift (Default)", "GnnLRP", "GradCam", "GNNExplainer"],
        help="At present, you can choose between 2 models (Flair or DistilBERT) to embed your text. More to come!",horizontal=True
    )
method=None
if ExplainerType=="DeepLift (Default)":
    method="deeplift"
elif ExplainerType=="GnnLRP":
    method="gnnlrp"
elif ExplainerType=="GradCam":
    method="gradcam"
elif ExplainerType=="GNNExplainer":
    method="gnnexplainer"


ModelType = st.sidebar.radio(
        "Choose your Model",
        ["GCN", "GIN"],
        help="GCN: Graph Convolutional Neural Network, GIN: Graph Isomorphic Network",horizontal=True
)

Vis_Style = st.sidebar.radio(
        "Choose your Visualization Style",
        ["Rdkit", "Custom"],
        help="Use cusom for GNNLrp or rdkit otherwise",horizontal=True
)


option=None
option = st.sidebar.selectbox('Select from Tox21 Database',tuple(list(tox_21_explanations.keys())[:60]))
if st.sidebar.button("process Selected tox"):
    #relevances_scale = create_relevance_from_data(tox_21_explanations[option]['data'], method = method)
    data = process_out(option)
    exs , x_collector = from_smiles(option, data_in="tox",model_type=ModelType,method=method)
    relevances_scale = create_relevance_from_explanation(data[0],exs ,method = method)

    if len(relevances_scale)!=0:
        actual="Non Toxic"
        if math.isnan(tox_21_explanations[option]['data'].y[0].item()):
            actual == None
        else:
            if int(tox_21_explanations[option]['data'].y[0].item())==1:
                actual = "Toxic"
        plot_streamlit(option, relevances_scale, actual=actual, x_collector = x_collector,data_source="tox",model_type=ModelType, data = data[0])
    else:
        st.markdown('<p style="font-family:sans-serif; color:red; font-size: 20px;">Explanation Not Available</p>', unsafe_allow_html=True)

user_input = st.sidebar.text_input("Provide Custom Smile: ", option)
if st.sidebar.button("process tox"):
    print("relavence from explanation called from Tox")
    data = process_out(user_input)
    if len(data)!=0:
        exs , x_collector = from_smiles(user_input, data_in="tox",model_type=ModelType,method=method)
        relevances_scale = create_relevance_from_explanation(data[0],exs ,method = method)
        plot_streamlit(user_input, relevances_scale,x_collector = x_collector, data_source="tox",model_type=ModelType, data=data[0])
    else:
        st.markdown('<p style="font-family:sans-serif; color:red; font-size: 50px;">Invalid Smile String</p>', unsafe_allow_html=True)

# clintox
option2 = st.sidebar.selectbox('Select from Clintox Database',tuple(list(clintox_explanations.keys())[:60]))

if st.sidebar.button("process Selected Clintox"):
    data = process_out(option2)
    exs , x_collector = from_smiles(option2,data_in="clintox",model_type=ModelType, method=method)
    relevances_scale = create_relevance_from_explanation(data[0],exs ,method = method)    
    if len(relevances_scale)!=0:
        actual="FDA Rejected"
        if math.isnan(clintox_explanations[option2]['data'].y[0].item()):
            actual == None
        else:
            if int(clintox_explanations[option2]['data'].y[0].item())==1:
                actual = "FDA Approved"
        plot_streamlit(option2, relevances_scale,actual=actual,x_collector = x_collector, data_source="clintox",model_type=ModelType, data=data[0])
    else:
        st.markdown('<p style="font-family:sans-serif; color:red; font-size: 20px;">Explanation Not Available</p>', unsafe_allow_html=True)

user_input = st.sidebar.text_input("Provide Custom Smile: ", option2)
if st.sidebar.button("process clintox"):
    data = process_out(user_input)
    if len(data)!=0:
        exs, x_collector = from_smiles(user_input,data_in="clintox",model_type=ModelType, method=method)
        relevances_scale = create_relevance_from_explanation(data[0],exs ,method = method)
        plot_streamlit(user_input, relevances_scale,x_collector = x_collector, data_source="clintox",model_type=ModelType, data=data[0])
    else:
        st.markdown('<p style="font-family:sans-serif; color:red; font-size: 50px;">Invalid Smile String</p>', unsafe_allow_html=True)

#O=C(O)Cc1cc(I)c(Oc2ccc(O)c(I)c2)c(I)c1
#test for rdkit vis
