import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from gurobipy import Model, GRB, quicksum, LinExpr
import eat
import graphviz
from IPython.display import Image, display
import pickle
import os
from scipy.stats import ks_2samp
import time

N1=list(range(0,100))
N2=list(range(100, 150))
R1, R2=[0], [1,2]
N1_bar=list(range(0,150))
N2_bar=list(range(0,100))
m, q, n = 3, 3, 150


def TreeDMUs(tree):
    res = {}
    for node in tree.tree:
        if node.get("SL") == -1:
            res[node["id"]] = {"dmus":node["index"],
                               'a': node['a'],
                               'b': node['b'],
                               'y': node['y']}
    return res

def stats(df, col):
    avg_eff = df[col].mean()
    min_eff = df[col].min()
    max_eff = df[col].max()
    std_dev_eff = df[col].std()
    num_eff = (df[col]==1).sum()
    num_considered = len(df)

    result = {
        'avg': avg_eff,
        'min': min_eff,
        'max': max_eff,
        'std_dev': std_dev_eff,
        'num_eff': num_eff,
        'num_considered': num_considered
    }
    return result

os.makedirs("./iterations", exist_ok=True)


s = time.time()
for i in range(10):
        iter_dir = f'./iterations/iter_{i}'
        tree_dir = os.path.join(iter_dir, 'tree')
        os.makedirs(iter_dir, exist_ok=True)
        os.makedirs(tree_dir, exist_ok=True)
        
        #######################
        Xa = np.random.choice(np.concatenate([np.arange(-50, 0), np.arange(1, 51)]), size=(100, 3))
        Xb = np.random.choice(np.concatenate([np.arange(-50, 0), np.arange(1, 51)]), size=(50, 3))

        Ya = np.random.choice(np.concatenate([np.arange(-50, 0), np.arange(1, 51)]), size=(100, 3))
        Yb = np.random.choice(np.concatenate([np.arange(-50, 0), np.arange(1, 51)]), size=(50, 1))

        dataA = np.concatenate([Xa, Ya], axis=1)
        dataB = np.concatenate([Xb, Yb], axis=1)

        columns_A = ['x1', 'x2', 'x3', 'y1', 'y2', 'y3']
        columns_B = ['x1', 'x2', 'x3', 'y1']

        dfA = pd.DataFrame(dataA, columns=columns_A)
        dfB = pd.DataFrame(dataB, columns=columns_B)

        dfA['DMU'] = np.arange(dfA.shape[0])
        dfB['DMU'] = np.arange(dfB.shape[0])
        #######################


        #######################
        tree1 = eat.EAT(dfA, ['x1', 'x2', 'x3'], ['y1', 'y2', 'y3'], numStop=10, fold=5)
        tree2 = eat.EAT(dfB, ['x1', 'x2', 'x3'], ['y1'], numStop=10, fold=5)

        treeleaf1 = TreeDMUs(tree1)
        treeleaf2 = TreeDMUs(tree2)
        
        nodes1, a1, d_T1 = {}, [], []
        for idx, leaf in enumerate(treeleaf1):
            nodes1[idx] = treeleaf1[leaf]['dmus'].tolist()
            a1.append(treeleaf1[leaf]['a'])
            d_T1.append(treeleaf1[leaf]['y'])
        a1 = np.array(a1)
        d_T1 = np.array(d_T1)

        nodes2, a2, d_T2 = {}, [], []
        for idx, leaf in enumerate(treeleaf2):
            nodes2[idx] = treeleaf2[leaf]['dmus'].tolist()
            a2.append(treeleaf2[leaf]['a'])
            d_T2.append(treeleaf2[leaf]['y'])
        a2 = np.array(a2)
        d_T2 = np.array(d_T2)
        #######################
        with open(os.path.join(tree_dir, 'nodes1.pkl'), 'wb') as f:
            pickle.dump(nodes1, f)
        with open(os.path.join(tree_dir, 'nodes2.pkl'), 'wb') as f:
            pickle.dump(nodes2, f)

        np.save(os.path.join(tree_dir, 'a1.npy'), a1)
        np.save(os.path.join(tree_dir, 'd_T1.npy'), d_T1)
        np.save(os.path.join(tree_dir, 'a2.npy'), a2)
        np.save(os.path.join(tree_dir, 'd_T2.npy'), d_T2)

        #######################
        dfa_selected = dfA[['x1', 'x2', 'x3']]
        dfb_selected = dfB[['x1', 'x2', 'x3']]
        x = np.concatenate([dfa_selected.values, dfb_selected.values], axis=0)
        dfa_selected = dfA[['y1', 'y2', 'y3']]
        dfb_selected = np.concatenate([dfB[['y1']].values, np.zeros((50, 2))], axis=1)
        y = np.concatenate([dfa_selected, dfb_selected], axis=0)
        #######################
        excel_path = os.path.join(iter_dir, 'data.xlsx')
        with pd.ExcelWriter(excel_path) as writer:
            pd.DataFrame(x, columns=['x1', 'x2', 'x3']).to_excel(writer, sheet_name='x', index=False)
            pd.DataFrame(y, columns=['y1', 'y2', 'y3']).to_excel(writer, sheet_name='y', index=False)



        Efficiencies = {}


        #######################
        alphas=[]
        betas=[]
        Results=[]
        lambdas_1=[]

        for o in N1:
            alpha = {}
            beta = {}
            lambdas = {}
            model = Model("DMU_Optimization")
            for i in range(m):
                for j in N1:
                    alpha[i, j] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.6, ub=0.85, name=f"alpha_{i}_{j}_0")
            beta[o, 0] = model.addVar(vtype=GRB.CONTINUOUS, lb=0,ub=1, name=f"beta_{j}_0")
            beta[o, 1] = model.addVar(vtype=GRB.CONTINUOUS,lb=0,ub=1, name=f"beta_{j}_1")

            for j in N1_bar:
                lambdas[j, 0] = model.addVar(vtype=GRB.CONTINUOUS,lb=0, ub=1, name=f"lambda_{j}_0")
            for j in N2_bar:
                lambdas[j, 1] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"lambda_{j}_1")

        
            model.setParam('DualReductions', 0)
            model.setObjective(
        (beta[o, 0] + beta[o,1])/2, GRB.MAXIMIZE
        )
        
            
            R_minus = [x[o, i] - min(x[:, i]) for i in range(m)]
            R_plus = [max(y[:, r]) - y[o, r] for r in range(q)]
            

            model.addConstr(
                    quicksum(lambdas[j, 0] * y[j,0] for j in N1_bar) >= y[o,0] + beta[o, 0] * R_plus[0]
                )
            model.addConstr(
                    quicksum(lambdas[j, 1] * y[j,1] for j in N2_bar) >= y[o,1] + beta[o, 1] * R_plus[1]
                )
            model.addConstr(
                    quicksum(lambdas[j, 1] * y[j,2] for j in N2_bar) >= y[o,2] + beta[o, 1] * R_plus[2]
                )
            

            for i in range(m):  

                model.addConstr(
                        quicksum(lambdas[j, 0] * alpha[i, j] * x[j,i] for j in N1) + quicksum(lambdas[j, 0] * x[j,i] for j in N2)
                        <= alpha[i, o] * x[o,i] - beta[o, 0] * alpha[i, o] * R_minus[i]
                    )
                
                model.addConstr(
                        quicksum(lambdas[j, 1] * (1-alpha[i, j]) * x[j,i] for j in N1)
                        <= (1-alpha[i, o]) * x[o,i] - beta[o, 1] * (1-alpha[i, o]) * R_minus[i]
                    )
                
            model.addConstr(quicksum(lambdas[j, 0] for j in N1_bar) == 1)
            model.addConstr(quicksum(lambdas[j, 1] for j in N2_bar) == 1)

            model.optimize()

            if model.status == GRB.OPTIMAL:
                alpha_values1 = model.getAttr("x", alpha)
                beta_values1 = model.getAttr("x", beta)
                lambda_values1 = model.getAttr("x", lambdas)
                for key, value in alpha_values1.items():
                    if key[1]==o:
                            alphas.append({
                                    "DMU": o,
                                    "alphas_RDM": {'key':key, 'value': value}
                                })
                for key,value in lambda_values1.items():
                    lambdas_1.append({
                                    "DMU": o,
                                    "lambda": {'key':key, 'value': value}
                                })
                for key, value in beta_values1.items():
                    if key[0]==o:
                        betas.append({
                                    "DMU": o,
                                    "betas": {'key':key, 'value': value}
                                })
                        
                objective_value = model.objVal
                Results.append({"DMU":o, "efficiency_RDM":1-objective_value})
            else:
                print("No optimal solution found.")


        dmu_values = {}
        for item in alphas:
            dmu = item['DMU']
            value = item['alphas_RDM']['value']
            dmu_values.setdefault(dmu, []).append(value)
        values_list = list(dmu_values.values())



        for o in N2:
            alpha = {}
            beta = {}
            lambdas = {}
            model = Model("DMU_Optimization")
            for i in range(m):
                for j in N1:
                    alpha[i, j] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.6, ub=0.85, name=f"alpha_{i}_{j}_0")
            beta[o, 0] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1,name=f"beta_{j}_0")

            for j in N1_bar:
                lambdas[j, 0] = model.addVar(vtype=GRB.CONTINUOUS,lb=0, ub=1,name=f"lambda_{j}_0")

            model.setParam('DualReductions', 0)
            model.setObjective(
        (beta[o, 0]), GRB.MAXIMIZE
        )
            
            R_minus = [x[o, i] - min(x[:, i]) for i in range(m)]
            R_plus = [max(y[:, r]) - y[o, r] for r in range(q)]

            model.addConstr(
                    quicksum(lambdas[j, 0] * y[j,0] for j in N1_bar) >= y[o,0] + beta[o, 0] * R_plus[0]
                )
            
            for i in range(m):  

                model.addConstr(
                        quicksum(lambdas[j, 0] * values_list[j][i] * x[j,i] for j in N1) + quicksum(lambdas[j, 0] * x[j,i] for j in N2)
                        <=  x[o,i] - beta[o, 0]* R_minus[i]
                    )
                
            model.addConstr(quicksum(lambdas[j, 0] for j in N1_bar) == 1)


        
            
            model.optimize()

            if model.status == GRB.OPTIMAL:
                alpha_values1 = model.getAttr("x", alpha)
                beta_values1 = model.getAttr("x", beta)
                lambda_values1 = model.getAttr("x", lambdas)
                for key, value in alpha_values1.items():
                    if key[1]==o:
                            alphas.append({
                                    "DMU": o,
                                    "alphas": {'key':key, 'value': value}
                                })
                for key,value in lambda_values1.items():
                    lambdas_1.append({
                                    "DMU": o,
                                    "lambda": {'key':key, 'value': value}
                                })
                for key, value in beta_values1.items():
                    if key[0]==o:
                        betas.append({
                                    "DMU": o,
                                    "betas": {'key':key, 'value': value}
                                })
                        
                objective_value = model.objVal
                Results.append({"DMU":o, "efficiency_RDM":1-objective_value})
            else:
                print("No optimal solution found.")
        Efficiencies['RDM'] = Results
        # RDM FDH
        Nalphas=[]
        betas=[]
        Results=[]
        lambdas_1=[]
        alphas = []
        for o in N1:
            alpha = {}
            beta = {}
            lambdas = {}
            model = Model("DMU_Optimization")
            for i in range(m):
                for j in N1:
                    alpha[i, j] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.6, ub=0.85, name=f"alpha_{i}_{j}_0")
            beta[o, 0] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"beta_{j}_0")
            beta[o, 1] = model.addVar(vtype=GRB.CONTINUOUS,  lb=0, ub=1,name=f"beta_{j}_1")

            for j in N1_bar:
                lambdas[j, 0] = model.addVar(vtype=GRB.BINARY,lb=0, ub=1, name=f"lambda_{j}_0")
            for j in N2_bar:
                lambdas[j, 1] = model.addVar(vtype=GRB.BINARY, lb=0, ub=1, name=f"lambda_{j}_1") 

        
            model.setParam('DualReductions', 0)
            model.setObjective(
        (beta[o, 0] + beta[o,1])/2, GRB.MAXIMIZE
        )
            
            R_minus = [x[o, i] - min(x[:, i]) for i in range(m)]
            R_plus = [max(y[:, r]) - y[o, r] for r in range(q)]
            
            model.addConstr(
                    quicksum(lambdas[j, 0] * y[j,0] for j in N1_bar) >= y[o,0] + beta[o, 0] * R_plus[0]
                )
            model.addConstr(
                    quicksum(lambdas[j, 1] * y[j,1] for j in N2_bar) >= y[o,1] + beta[o, 1] * R_plus[1]
                )
            model.addConstr(
                    quicksum(lambdas[j, 1] * y[j,2] for j in N2_bar) >= y[o,2] + beta[o, 1] * R_plus[2]
                )
        

            for i in range(m):  

                model.addConstr(
                        quicksum(lambdas[j, 0] * alpha[i, j] * x[j,i] for j in N1) + quicksum(lambdas[j, 0] * x[j,i] for j in N2)
                        <= alpha[i, o] * x[o,i] - beta[o, 0] * alpha[i, o] * R_minus[i]
                    )
                model.addConstr(
                        quicksum(lambdas[j, 1] * (1-alpha[i, j]) * x[j,i] for j in N1)
                        <= (1-alpha[i, o]) * x[o,i] - beta[o, 1] * (1-alpha[i, o]) * R_minus[i]
                    )
                
            model.addConstr(quicksum(lambdas[j, 0] for j in N1_bar) == 1)
            model.addConstr(quicksum(lambdas[j, 1] for j in N2_bar) == 1)

            model.optimize()

            if model.status == GRB.OPTIMAL:
                alpha_values1 = model.getAttr("x", alpha)
                beta_values1 = model.getAttr("x", beta)
                lambda_values1 = model.getAttr("x", lambdas)
                for key, value in alpha_values1.items():
                    if key[1]==o:
                            alphas.append({
                                    "DMU": o,
                                    "alphas_RDM_FDH": {'key':key, 'value': value}
                                })
                for key,value in lambda_values1.items():
                    lambdas_1.append({
                                    "DMU": o,
                                    "lambda": {'key':key, 'value': value}
                                })
                for key, value in beta_values1.items():
                    if key[0]==o:
                        betas.append({
                                    "DMU": o,
                                    "betas": {'key':key, 'value': value}
                                })
                        
                objective_value = model.objVal
                Results.append({"DMU":o, "efficiency_RDM_FDH":1-objective_value})
            else:
                print("No optimal solution found.")

        dmu_values = {}
        for item in alphas:
            print(item)
            dmu = item['DMU']
            value = item['alphas_RDM_FDH']['value']
            dmu_values.setdefault(dmu, []).append(value)
        values_list = list(dmu_values.values())


        for o in N2:
            alpha = {}
            beta = {}
            lambdas = {}
            model = Model("DMU_Optimization")
            for i in range(m):
                for j in N1:
                    alpha[i, j] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.1, ub=0.4, name=f"alpha_{i}_{j}_0")
            beta[o, 0] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"beta_{j}_0")

            for j in N1_bar:
                lambdas[j, 0] = model.addVar(vtype=GRB.BINARY,lb=0, ub=1, name=f"lambda_{j}_0")  

        
            model.setParam('DualReductions', 0)
            model.setObjective(
        (beta[o, 0]), GRB.MAXIMIZE
        )
            
            R_minus = [x[o, i] - min(x[:, i]) for i in range(m)]
            R_plus = [max(y[:, r]) - y[o, r] for r in range(q)]

            model.addConstr(
                    quicksum(lambdas[j, 0] * y[j,0] for j in N1_bar) >= y[o,0] + beta[o, 0] * R_plus[0]
                )

            for i in range(m):  
                model.addConstr(
                        quicksum(lambdas[j, 0] * values_list[j][i] * x[j,i] for j in N1) + quicksum(lambdas[j, 0] * x[j,i] for j in N2)
                        <=  x[o,i] - beta[o, 0]* R_minus[i]
                    )
                
            model.addConstr(quicksum(lambdas[j, 0] for j in N1_bar) == 1)
            
            model.optimize()

            if model.status == GRB.OPTIMAL:
                alpha_values1 = model.getAttr("x", alpha)
                beta_values1 = model.getAttr("x", beta)
                lambda_values1 = model.getAttr("x", lambdas)
                for key, value in alpha_values1.items():
                    if key[1]==o:
                            alphas.append({
                                    "DMU": o,
                                    "alphas": {'key':key, 'value': value}
                                })
                for key,value in lambda_values1.items():
                    lambdas_1.append({
                                    "DMU": o,
                                    "lambda": {'key':key, 'value': value}
                                })
                for key, value in beta_values1.items():
                    if key[0]==o:
                        betas.append({
                                    "DMU": o,
                                    "betas": {'key':key, 'value': value}
                                })
                        
                objective_value = model.objVal
                Results.append({"DMU":o, "efficiency_RDM_FDH":1-objective_value})

            else:
                print("No optimal solution found.")
        Efficiencies['RDMFDH'] = Results
        #EAT FDH
        alphas=[]
        betas=[]
        Results=[]
        lambdas_1=[]

        for o in N1:
            alpha = {}
            beta = {}
            lambdas = {}
            lambdas_dash={}
            model = Model("DMU_Optimization")
            for t in nodes1:
                alpha[t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.6, ub=0.85, name=f"alpha_{t}_0")

            beta[o, 0] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"beta_{o}_0")
            beta[o, 1] = model.addVar(vtype=GRB.CONTINUOUS,  lb=0,name=f"beta_{o}_1")

            for t in nodes1:
                lambdas[t, 0] = model.addVar(vtype=GRB.BINARY, name=f"lambda_{t}_0")
            for t in nodes2:
                lambdas_dash[t, 0] = model.addVar(vtype=GRB.BINARY, name=f"lambda_dash_{t}_0")
            for t in nodes1:
                lambdas[t, 1] = model.addVar(vtype=GRB.BINARY, name=f"lambda_{t}_1")


        
            model.setParam('DualReductions', 0)
            model.setObjective(
        (beta[o, 0] + beta[o,1])/2, GRB.MAXIMIZE
        )
            
            R_minus = [x[o, i] - min(x[:, i]) for i in range(m)]
            R_plus = [max(y[:, r]) - y[o, r] for r in range(q)]
            
            model.addConstr(
                                quicksum(lambdas[t1, 0] * d_T1[t1,0] for t1 in nodes1) + quicksum(lambdas_dash[t2, 0] * d_T2[t2][0] for t2 in nodes2) >= y[o,0] + beta[o, 0] * R_plus[0]
                            )
                    
            model.addConstr(
                                quicksum(lambdas[t, 1] * d_T1[t,1] for t in nodes1) >= y[o,1] + beta[o, 1] * R_plus[1]
                            )   
            model.addConstr(
                                quicksum(lambdas[t, 1] * d_T1[t,2] for t in nodes1) >= y[o,2] + beta[o, 1] * R_plus[2]
                            )  
            
            alpha_knot = alpha[next((key for key, values in nodes1.items() if o in values), None)]
            for i in range(m):  

                    model.addConstr(
                            quicksum(lambdas[t, 0] * alpha[t] * a1[t,i] for t in nodes1) + quicksum(lambdas_dash[t, 0] * a2[t,i] for t in nodes2)
                            <= alpha_knot * x[o,i] - beta[o, 0] * alpha_knot * R_minus[i]
                        )
                    model.addConstr(
                            quicksum(lambdas[t, 1] * (1-alpha[t]) * a1[t,i] for t in nodes1)
                            <= (1-alpha_knot) * x[o,i] - beta[o, 1] * (1-alpha_knot) * R_minus[i]
                        )
                    
            model.addConstr(quicksum(lambdas[t, 0] for t in nodes1)+ quicksum(lambdas_dash[t, 0] for t in nodes2)== 1)
            model.addConstr(quicksum(lambdas[t, 1] for t in nodes1)== 1)
            
            model.optimize()

            if model.status == GRB.OPTIMAL:
                alpha_values1 = model.getAttr("x", alpha)
                beta_values1 = model.getAttr("x", beta)
                lambda_values1 = model.getAttr("x", lambdas)
                for key, value in alpha_values1.items():
                    if key==next((key for key, values in nodes1.items() if o in values), None):
                            alphas.append({
                                                "DMU": o,
                                                "alphas": {'key':key, 'value': value}
                                            })
                for key,value in lambda_values1.items():
                    lambdas_1.append({
                                    "DMU": o,
                                    "lambda": {'key':key, 'value': value}
                                })
                for key, value in beta_values1.items():
                    if key[0]==o:
                        betas.append({
                                    "DMU": o,
                                    "betas": {'key':key, 'value': value}
                                })
                        
                objective_value = model.objVal
                Results.append({"DMU":o, "efficiency_EAT_FDH":
                                1-objective_value})
            else:
                print("No optimal solution found.")





        alpha_node=[]
        from collections import defaultdict
        alpha_sums = defaultdict(float)
        alpha_counts = defaultdict(int)
        def calculate_average_alphas(data):
            for item in data:
                key = item['alphas']['key']
                value = item['alphas']['value']
                alpha_sums[key] += value
                alpha_counts[key] += 1

            average_alphas = {key: alpha_sums[key] / alpha_counts[key] for key in alpha_sums}
            for key in average_alphas:
                alpha_node.append(average_alphas[key])
            return alpha_node
        alpha_node = calculate_average_alphas(alphas)
        print(alpha_node)




        for o in N2:
            alpha = {}
            beta = {}
            lambdas = {}
            model = Model("DMU_Optimization")
            beta[o, 0] = model.addVar(vtype=GRB.CONTINUOUS, lb=0,name=f"beta_{o}_0")

            for t in nodes1:
                lambdas[t, 0] = model.addVar(vtype=GRB.BINARY,lb=0,ub=1, name=f"lambda_{t}_0")
            for t in nodes2:
                lambdas_dash[t, 0] = model.addVar(vtype=GRB.BINARY, lb=0,ub=1,name=f"lambda_dash_{t}_0")


        
            model.setParam('DualReductions', 0)
            model.setObjective(
        (beta[o, 0]), GRB.MAXIMIZE
        )
            
            
            R_minus = [x[o, i] - min(x[:, i]) for i in range(m)]
            R_plus = [max(y[:, r]) - y[o, r] for r in range(q)]
            
            model.addConstr(
                    quicksum(lambdas[t, 0] * d_T1[t,0] for t in nodes1)+quicksum(lambdas_dash[t, 0] * d_T2[t,0] for t in nodes2) >= y[o,0] + beta[o, 0] * R_plus[0]
                )
            
            for i in range(m):  
                
            
                model.addConstr(
                        quicksum(lambdas[t, 0] *alpha_node[t]*a1[t,i] for t in nodes1) + quicksum(lambdas_dash[t, 0] * a2[t,i] for t in nodes2)
                        <=  x[o,i] - beta[o, 0]* R_minus[i]
                    )
                
            model.addConstr(quicksum(lambdas[t, 0] for t in nodes1)+ quicksum(lambdas_dash[t, 0] for t in nodes2)== 1)

            model.optimize()

            if model.status == GRB.OPTIMAL:
                alpha_values1 = model.getAttr("x", alpha)
                beta_values1 = model.getAttr("x", beta)
                lambda_values1 = model.getAttr("x", lambdas)
                for key, value in alpha_values1.items():
                            alphas.append({
                                    "DMU": o,
                                    "alphas": {'key':key, 'value': value}
                                })
                
                
                for key,value in lambda_values1.items():
                    lambdas_1.append({
                                    "DMU": o,
                                    "lambda": {'key':key, 'value': value}
                                })
                for key, value in beta_values1.items():
                    if key[0]==o:
                        betas.append({
                                    "DMU": o,
                                    "betas": {'key':key, 'value': value}
                                })
                        
                objective_value = model.objVal
                Results.append({"DMU":o, "efficiency_EAT_FDH":1-objective_value})

            
            else:
                print("No optimal solution found.")

        Efficiencies['EATFDH'] = Results
        # EAT

        alphas=[]
        betas=[]
        Results=[]
        lambdas_1=[]

        for o in N1:
            alpha = {}
            beta = {}
            lambdas = {}
            lambdas_dash={}
            model = Model("DMU_Optimization")
            for t in nodes1:
                alpha[t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.6, ub=0.85, name=f"alpha_{t}_0")

            beta[o, 0] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"beta_{o}_0")
            beta[o, 1] = model.addVar(vtype=GRB.CONTINUOUS,  lb=0,name=f"beta_{o}_1")

            for t in nodes1:
                lambdas[t, 0] = model.addVar(vtype=GRB.CONTINUOUS,lb=0, name=f"lambda_{t}_0")
            for t in nodes2:
                lambdas_dash[t, 0] = model.addVar(vtype=GRB.CONTINUOUS,lb=0, name=f"lambda_dash_{t}_0")
            for t in nodes1:
                lambdas[t, 1] = model.addVar(vtype=GRB.CONTINUOUS,lb=0, name=f"lambda_{t}_1")

        
            model.setParam('DualReductions', 0)
            model.setObjective(
        (beta[o, 0] + beta[o,1])/2, GRB.MAXIMIZE
        )
            
            R_minus = [x[o, i] - min(x[:, i]) for i in range(m)]
            R_plus = [max(y[:, r]) - y[o, r] for r in range(q)]
            
            model.addConstr(
                                quicksum(lambdas[t, 0] * d_T1[t,0] for t in nodes1) + quicksum(lambdas_dash[t, 0] * d_T2[t, 0] for t in nodes2) >= y[o,0] + beta[o, 0] * R_plus[0]
                            )
                    
            model.addConstr(
                                quicksum(lambdas[t, 1] * d_T1[t,1] for t in nodes1) >= y[o,1] + beta[o, 1] * R_plus[1]
                            )   
            model.addConstr(
                                quicksum(lambdas[t, 1] * d_T1[t,2] for t in nodes1) >= y[o,2] + beta[o, 1] * R_plus[2]
                            )  
            
            alpha_knot = alpha[next((key for key, values in nodes1.items() if o in values), None)]
            for i in range(m):  

                    model.addConstr(
                            quicksum(lambdas[t, 0] * alpha[t] * a1[t,i] for t in nodes1) + quicksum(lambdas_dash[t, 0] * a2[t,i] for t in nodes2)
                            <= alpha_knot * x[o,i] - beta[o, 0] * alpha_knot * R_minus[i]
                        )
                    model.addConstr(
                            quicksum(lambdas[t, 1] * (1-alpha[t]) * a1[t,i] for t in nodes1)
                            <= (1-alpha_knot) * x[o,i] - beta[o, 1] * (1-alpha_knot) * R_minus[i]
                        )
            model.addConstr(quicksum(lambdas[t, 0] for t in nodes1)+ quicksum(lambdas_dash[t, 0] for t in nodes2)== 1)
            model.addConstr(quicksum(lambdas[t, 1] for t in nodes1)== 1)

            model.optimize()

            if model.status == GRB.OPTIMAL:
                alpha_values1 = model.getAttr("x", alpha)
                beta_values1 = model.getAttr("x", beta)
                lambda_values1 = model.getAttr("x", lambdas)
                for key, value in alpha_values1.items():
                    if key==next((key for key, values in nodes1.items() if o in values), None):
                            alphas.append({
                                                "DMU": o,
                                                "alphas": {'key':key, 'value': value}
                                            })
                for key,value in lambda_values1.items():
                    lambdas_1.append({
                                    "DMU": o,
                                    "lambda": {'key':key, 'value': value}
                                })
                for key, value in beta_values1.items():
                    if key[0]==o:
                        betas.append({
                                    "DMU": o,
                                    "betas": {'key':key, 'value': value}
                                })
                        
                objective_value = model.objVal
                Results.append({"DMU":o, "efficiency_EAT":
                                1-objective_value})
            else:
                print("No optimal solution found.")
                


        alpha_node=[]
        from collections import defaultdict
        alpha_sums = defaultdict(float)
        alpha_counts = defaultdict(int)

        def calculate_average_alphas(data):
            for item in data:
                key = item['alphas']['key']
                value = item['alphas']['value']
                alpha_sums[key] += value
                alpha_counts[key] += 1

            average_alphas = {key: alpha_sums[key] / alpha_counts[key] for key in alpha_sums}
            for key in average_alphas:
                alpha_node.append(average_alphas[key])
            return alpha_node

        alpha_node = calculate_average_alphas(alphas)
        print(alpha_node)


        import pandas as pd
        for o in N2:
            alpha = {}
            beta = {}
            lambdas = {}
            model = Model("DMU_Optimization")
            beta[o, 0] = model.addVar(vtype=GRB.CONTINUOUS, lb=0,name=f"beta_{o}_0")

            for t in nodes1:
                lambdas[t, 0] = model.addVar(vtype=GRB.CONTINUOUS,lb=0, name=f"lambda_{t}_0")
            for t in nodes2:
                lambdas_dash[t, 0] = model.addVar(vtype=GRB.CONTINUOUS, lb=0,name=f"lambda_dash_{t}_0")


        
            model.setParam('DualReductions', 0)
            model.setObjective(
        (beta[o, 0]), GRB.MAXIMIZE
        )

            R_minus = [x[o, i] - min(x[:, i]) for i in range(m)]
            R_plus = [max(y[:, r]) - y[o, r] for r in range(q)]

            model.addConstr(
                    quicksum(lambdas[t, 0] * d_T1[t,0] for t in nodes1)+quicksum(lambdas_dash[t, 0] * d_T2[t,0] for t in nodes2) >= y[o,0] + beta[o, 0] * R_plus[0]
                )
            
            for i in range(m):  
                
            
                model.addConstr(
                        quicksum(lambdas[t, 0] *alpha_node[t]*a1[t,i] for t in nodes1) + quicksum(lambdas_dash[t, 0] * a2[t,i] for t in nodes2)
                        <=  x[o,i] - beta[o, 0]* R_minus[i]
                    )
                
            model.addConstr(quicksum(lambdas[t, 0] for t in nodes1)+ quicksum(lambdas_dash[t, 0] for t in nodes2)== 1)

        
            
            model.optimize()


            if model.status == GRB.OPTIMAL:
                alpha_values1 = model.getAttr("x", alpha)
                beta_values1 = model.getAttr("x", beta)
                lambda_values1 = model.getAttr("x", lambdas)
                for key, value in alpha_values1.items():
                            alphas.append({
                                    "DMU": o,
                                    "alphas": {'key':key, 'value': value}
                                })
                
                
                for key,value in lambda_values1.items():
                    lambdas_1.append({
                                    "DMU": o,
                                    "lambda": {'key':key, 'value': value}
                                })
                for key, value in beta_values1.items():
                    if key[0]==o:
                        betas.append({
                                    "DMU": o,
                                    "betas": {'key':key, 'value': value}
                                })
                        
                objective_value = model.objVal
                Results.append({"DMU":o, "efficiency_EAT":1-objective_value})

            
            else:
                print("No optimal solution found.")
        Efficiencies['EAT'] = Results
        #######################



        #######################
        dfres4 = pd.DataFrame(Efficiencies['EAT'])
        dfres3 = pd.DataFrame(Efficiencies['EATFDH'])
        dfres2 = pd.DataFrame(Efficiencies['RDMFDH'])
        dfres1 = pd.DataFrame(Efficiencies['RDM'])

        df_combined = pd.DataFrame()
        df_combined = pd.concat([
            dfres1,
            dfres2,
            dfres3,
            dfres4
        ], axis=1)
        print(df_combined.columns)
        df_combined = df_combined[['efficiency_RDM', 'efficiency_RDM_FDH', 'efficiency_EAT', 'efficiency_EAT_FDH']]
        #######################


        #######################
        dfstats1 = pd.DataFrame(list(stats(df_combined, 'efficiency_RDM').items()), columns=['Index', 'RDM'])
        dfstats2 = pd.DataFrame(list(stats(df_combined, 'efficiency_RDM_FDH').items()), columns=['Index', 'RDM_FDH'])
        dfstats3 = pd.DataFrame(list(stats(df_combined, 'efficiency_EAT').items()), columns=['Index', 'EAT'])
        dfstats4 = pd.DataFrame(list(stats(df_combined, 'efficiency_EAT_FDH').items()), columns=['Index', 'EAT_FDH'])

        df_merged = dfstats1.merge(dfstats2, on='Index', how='outer') \
                            .merge(dfstats3, on='Index', how='outer') \
                            .merge(dfstats4, on='Index', how='outer')
        #######################

        #######################
        dfeff = df_combined.copy()
        dfeff.rename(columns={'efficiency_EAT_FDH':'E_CRE',
                            'efficiency_EAT':'E_NCRE',
                            'efficiency_RDM_FDH':'E_NCR',
                            'efficiency_RDM':'E_CR'}, inplace=True)
        CrM = dfeff.iloc[:, 0:4].corr()
        #######################


        #######################
        pairs = [('E_CR', 'E_NCR'), ('E_CR', 'E_NCRE'), ('E_CR', 'E_CRE'),
                ('E_NCR', 'E_NCRE'), ('E_NCR', 'E_CRE'),
                ('E_NCRE', 'E_CRE')]

        sns.set(style="whitegrid")
        plt.figure(figsize=(10, 10), dpi=300)  

        for i, (var1, var2) in enumerate(pairs, 1):
            plt.subplot(2, 3, i)

            x = dfeff[f'{var1}']
            y = dfeff[f'{var2}']

            plt.scatter(x, y, label=f'{var1} vs {var2}', color='blue')
            plt.plot([0, 1], [0, 1], color='red', linestyle='--')  

            r = np.corrcoef(x, y)[0, 1]
            plt.text(0.05, 0.9, f'r = {r:.4f}', transform=plt.gca().transAxes,
                    fontsize=12, fontweight='bold', color='black',
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.xlabel(var1)
            plt.ylabel(var2)
            plt.title(f'{var1} vs {var2}')

            plt.gca().set_aspect('equal', adjustable='box')

        plt.tight_layout(pad=0.0)  

        save_path = os.path.join(iter_dir, 'pairwise_scatter.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        #######################
        # CEAT = E_CRE = EAT_FDH
        models = ['E_CR', 'E_NCR', 'E_NCRE', 'E_CRE']
        results = []

        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                M1 = models[i]
                M2 = models[j]
                data1 = dfeff[M1].dropna().values
                data2 = dfeff[M2].dropna().values
                
                ks_stat, p_value = ks_2samp(data1, data2)
                results.append([M1, M2, ks_stat, p_value])

        results_df = pd.DataFrame(results, columns=['Model1', 'Model2', 'TKS', 'p-value'])
        #######################

        df_combined.to_csv(os.path.join(iter_dir, 'efficiencies.csv'), index=False)
        df_merged.to_csv(os.path.join(iter_dir, 'stats.csv'), index=False)
        CrM.to_csv(os.path.join(iter_dir, 'CrM.csv'), index=False)
        results_df.to_csv(os.path.join(iter_dir, 'kstest.csv'), index=False)

print('timetaken:', time.time()-s)