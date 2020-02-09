import numpy as np
import numpy.random as npr
from scipy.stats import invwishart
from gs_real import main_sampling

# different g_prior independent g_prior for each time slice
'''
for prediction
'''
ob = np.load('coleman_train.npy')
print(len(np.where(ob==-1)[0]),'len(np.where(ob==-1)[0])')
print(len(np.where(ob>-1)[0]),'len(np.where(ob>-1)[0])')
tr0x,tr0y,tr0t = np.where(ob==0)
tr1x,tr1y,tr1t = np.where(ob==1)

d_num = np.shape(ob)[0]# data number
tspan = np.shape(ob)[2]# timeslot    
tspan_fcp = (tspan-1)*2+1
print(d_num,'d_num')
auc = np.zeros((d_num,d_num,tspan))
obr = np.load('coleman.npy')
mc_num = 200 # max_cluster_number

b_mu_ll = 2 # B_ll mean 
b_v = 5 # B_ll variance
b_mu_lk = -3 # B_lk mean 
q_mu = -3 # sparsity parameter mean 
q_v = 5 # sparsity parameter variance
epsilon = -10

ivw_df = 5 #inverse wishart df
ivw_scale = np.array(([1,0],[0,1])) # inverse wishart scale matrix

burn_iter = 50
sample_iter = 100

g_num = 3 # group number
g_prior = np.ones(tspan)*0.1 # group piror

f_para = 3 # fragmentation parameter
c_para = 3 # coagulation parameter


print(g_prior,'g_prior',g_num,'g_num')
g_mat = np.zeros((d_num,g_num,tspan)) #group membership
g_ij = np.zeros((d_num,d_num,tspan,2),dtype = int)-1 #group member indicator
z_i = np.zeros((d_num,tspan_fcp),dtype=int)-1 #cluster indicator
fcp = np.zeros((mc_num,mc_num,tspan_fcp-1)) #transition table
fcp_table = np.zeros((mc_num,tspan_fcp)) #record cluster status 1:existing -1:new 0:null
b_mat_mu = np.zeros((g_num,g_num)) # B parameter 
b_mat_v = np.zeros((2,2,g_num,g_num)) # B parameter variance low triangle equal to 0 
q_arr = np.zeros(g_num) # sparsity parameter


'''
initial
'''

        

z_i[0,:] = 0
fcp_table[0,:] = 1
fcp[0,0,:] = 1

for i in range(1, d_num):
    for t in range(tspan_fcp):
#             for t in range(1):
        if t%2 == 0: # fragmentation
            if t == 0:
                entity_index = np.where(z_i[:,t]>=0)[0]
                c_index, c_count = np.unique(z_i[entity_index,t], return_counts = True)
                empty_c_index = np.where(fcp_table[:,t]==0)[0]
                empty_c = empty_c_index[0]
                c_index = np.append(c_index, empty_c)
                c_count = np.append(c_count, f_para)
                p = c_count/np.sum(c_count)
                index = np.argmax(npr.multinomial(1,p))
                fcp_table[c_index[index],t] = 1
                z_i[i,t] = c_index[index]
            else:
                prev_c = z_i[i,t-1]
                entity_index = np.where(z_i[:,t-1]==prev_c)[0]
                i_index = np.where(entity_index==i)[0]
                entity_index = np.delete(entity_index, i_index)
                if len(entity_index)==0:
                    empty_c_index = np.where(fcp_table[:,t]==0)[0]
                    empty_c = empty_c_index[0]
                    z_i[i,t] = empty_c  
                    fcp_table[empty_c,t] = 1
                    fcp[prev_c,empty_c] = fcp[prev_c,empty_c]+1
                else:
                    c_index, c_count = np.unique(z_i[entity_index,t], return_counts = True)
                    empty_c_index = np.where(fcp_table[:,t]==0)[0]
                    empty_c = empty_c_index[0]
                    c_index = np.append(c_index, empty_c)
                    c_count = np.append(c_count, f_para)
                    p = c_count/np.sum(c_count)
                    index = np.argmax(npr.multinomial(1,p))
                    fcp_table[c_index[index],t] = 1
                    fcp[prev_c,c_index[index],t-1] = fcp[prev_c,c_index[index],t-1] + 1
                    z_i[i,t] = c_index[index]
                
        else: # coagulation
            prev_c = z_i[i,t-1]
            entity_index = np.where(z_i[:,t-1]==prev_c)[0]
            i_index = np.where(entity_index==i)[0]
            entity_index = np.delete(entity_index, i_index)
            if len(entity_index)==0:
                
                c_index = np.where(fcp_table[:,t]>0)[0]
                empty_c_index = np.where(fcp_table[:,t]==0)[0]
                empty_c = empty_c_index[0]
                c_count = np.zeros(len(c_index)+1)
                c_count[-1] = c_para
                for j in range(len(c_index)):
                    c_index_array = np.where(fcp[c_index[j],:,t-1]>0)[0]
                    c_count[j] = len(c_index_array)
                c_index = np.append(c_index, empty_c)
                p = c_count/np.sum(c_count)
                index = np.argmax(npr.multinomial(1,p))
                fcp_table[c_index[index],t] = 1
                fcp[prev_c,index,t-1] = fcp[prev_c,index,t-1] + 1
                z_i[i,t] = c_index[index]   
            else:
                c = z_i[entity_index[0],t]
                z_i[i,t] = c 
                fcp[prev_c,c,t-1] = fcp[prev_c,c,t-1]+1


#             print(np.sum(fcp),'fcp')
'''
initialize fcp fcp_table z_i
'''

for t in range(tspan):
    for i in range(d_num):
        for j in range(d_num):
            if ob[i,j,t]>-1:
                g = npr.randint(g_num,size = 1)
                g_mat[i,g,t] = g_mat[i,g,t]+1
                g_ij[i,j,t,0] = g
                g = npr.randint(g_num,size = 1)
                g_mat[j,g,t] = g_mat[j,g,t]+1
                g_ij[i,j,t,1] = g

#             print(np.sum(g_mat),'g_mat')
#             print(len(np.where(g_ij>-1)[0]),'g_ij')
'''
initialize g_mat g_ij
'''

for i in range(g_num):
    for j in range(i,g_num):
        if i==j:
            q_arr[i] = npr.normal(q_mu, np.sqrt(q_v))
            b_mat_mu[i,j] = npr.normal(b_mu_ll, np.sqrt(b_v))
        else:
            cov = invwishart.rvs(df = ivw_df,scale = ivw_scale)
            b_mat_v[:,:,i,j] = np.copy(cov)
#                         print(b_mat_v[:,:,i,j],i,j)
            mu = npr.multivariate_normal([b_mu_lk,b_mu_lk],cov)
            b_mat_mu[i,j] = np.copy(mu[0])
            b_mat_mu[j,i] = np.copy(mu[1])
        
'''
initialize B,q
'''

'''
main sampling
'''
main_sampling(sample_iter, d_num, tspan_fcp, z_i, fcp, fcp_table, ob, tspan,\
b_mat_mu, g_ij, q_mu, epsilon, mc_num, f_para, c_para, g_num, q_arr, b_mu_ll,\
b_v, q_v, b_mat_v, ivw_df, ivw_scale, b_mu_lk, g_mat, tr1x, tr1y, tr1t, tr0x,\
tr0y, tr0t, burn_iter, obr, g_prior)
    
    



