import numpy as np
import numpy.random as npr
from pypolyagamma import PyPolyaGamma
from numpy.linalg import inv
from scipy.stats import invwishart
from sklearn import metrics
from sklearn.metrics import roc_auc_score

# import matplotlib.pyplot as plt

def main_sampling(sample_iter, d_num, tspan_fcp, z_i, fcp, fcp_table, ob, tspan,\
    b_mat_mu, g_ij, q_mu, epsilon, mc_num, f_para, c_para, g_num, q_arr, b_mu_ll,\
    b_v, q_v, b_mat_v, ivw_df, ivw_scale, b_mu_lk, g_mat, tr1x, tr1y, tr1t, tr0x,\
    tr0y, tr0t, burn_iter, obr, g_prior):
    
    auc = np.zeros((d_num,d_num,tspan))
    likelog = np.zeros(sample_iter)
#     print(z_i,'z_i')
    b_mat_rebutall = np.zeros((g_num,g_num,sample_iter))
    q_rebutall = np.zeros((g_num,sample_iter))
    for ts in range(sample_iter):
        if ts%50 == 0:
            print(ts)
        for i in range(d_num):
            
            prob_mat = np.zeros((mc_num,mc_num,tspan_fcp-1)) # transition probability matrix
            p_c = np.zeros((mc_num,tspan_fcp)) #probability for each cluster
            pps_bf(i,z_i,tspan_fcp,fcp,fcp_table)
             
            for t in range(tspan_fcp):
                if t==0:
                    f_init_oper(p_c,fcp_table,z_i,f_para)
                    p_c[:,0] = p_ob(i,0,mc_num,g_ij,ob,fcp_table,b_mat_mu,q_arr,z_i,epsilon)*p_c[:,0]
 
                elif t%2==1:
                    prob_table = c_oper(t-1,mc_num,fcp_table,fcp,c_para)
                    prob_mat[:,:,t-1] = np.copy(prob_table)
                    p_c[:,t] = np.transpose(np.dot(np.transpose(p_c[:,t-1]),prob_table))
                else:
                    prob_table = f_oper(t-1,mc_num,fcp,fcp_table,f_para)
                    prob_mat[:,:,t-1] = np.copy(prob_table)
                    p_c[:,t] = np.transpose(np.dot(np.transpose(p_c[:,t-1]),prob_table))
                    p_c[:,t] = p_c[:,t]*p_ob(i,int(t/2),mc_num,g_ij,ob,fcp_table,b_mat_mu,q_arr,z_i,epsilon)
                normalize_p_c(p_c,t)
            s = npr.multinomial(1,p_c[:,tspan_fcp-1]/np.sum(p_c[:,tspan_fcp-1]))
            index_c = np.where(s==1)[0]
             

            z_i[i,tspan_fcp-1] = np.copy(index_c)
            fcp_table[index_c,tspan_fcp-1] = 1

            for t in range(tspan_fcp-2,-1,-1):
 
                s = prob_mat[:,index_c,t]*np.reshape(p_c[:,t],(mc_num,1))
                s = np.reshape(s,mc_num)

 
                s = npr.multinomial(1,s/np.sum(s))
                index_c = np.where(s==1)[0]
                z_i[i,t] = np.copy(index_c)
                fcp_table[index_c,t] = 1 
             
            x,y = np.where(fcp_table==-1)
            fcp_table[x,y] = 0
            s1 = np.copy(z_i[i,0:tspan_fcp-1])
            s2 = np.copy(z_i[i,1:tspan_fcp])
            fcp[s1,s2,np.arange(tspan_fcp-1)] = fcp[s1,s2,np.arange(tspan_fcp-1)]+1
            
#             print(z_i,'z_i')
#             print(fcp_table,'fcp_table')
            '''
            sample g_ij
            '''    
            for t in range(tspan):
                s_g_ij(i, t,d_num,g_num,ob,g_mat,g_ij,g_prior,z_i,b_mat_mu,q_arr,epsilon)
                
        s_blk(g_num,b_mu_ll,q_mu,b_v,q_v, b_mat_mu,q_arr,b_mu_lk,b_mat_v,ob,g_ij,z_i)
        s_b_var(g_num,b_mat_mu,ivw_scale,b_mat_v,ivw_df,b_mu_lk)
        b_mat_rebutall[:,:,ts] = np.copy(b_mat_mu)
        q_rebutall[:,ts] = np.copy(q_arr)
        '''
        sample b_lk
        '''
        pred = predict(g_mat,g_prior,g_num,b_mat_mu,q_arr,epsilon,d_num,tspan,z_i)
        likelog[ts] = np.sum(np.log(pred[tr1x,tr1y,tr1t]))+np.sum(np.log(1-pred[tr0x,tr0y,tr0t]))
        if ts>burn_iter:
            auc = pred+auc
    
#     print(likelog,'likelog')
    
    
    print(b_mat_mu,'b_mat_mu')
    print(q_arr,'q_arr')
#     x = np.arange(sample_iter)
#     plt.plot(x,likelog)
#     plt.show()
    np.save('likelog.npy',likelog)
    '''
    prediction
    '''
    
    
    
    
    x,y,t = np.where(ob>-1)
    our = np.copy(auc[x,y,t])
    cor = np.copy(obr[x,y,t])
    print('train auc',roc_auc_score(cor, our))
    x,y,t = np.where(ob==-1)
    our = np.copy(auc[x,y,t])
    cor = np.copy(obr[x,y,t])
    print('overall test auc',roc_auc_score(cor, our))
 
    

    '''
    save z_i
    print z_i
    '''

    np.save('z_i.npy',z_i)
    np.save('g_mat.npy',g_mat)
    np.save('auc.npy',auc)
# preprocess backward fordward --- delete all related z_i

def pps_bf(i,z_i,tspan_fcp,fcp,fcp_table):  
    index_1 = np.copy(z_i[i,0:tspan_fcp-1])
    index_2 = np.copy(z_i[i,1:tspan_fcp])
    fcp[index_1,index_2,np.arange(tspan_fcp-1)] = fcp[index_1,index_2,np.arange(tspan_fcp-1)]-1
    re = np.equal(fcp[index_1,index_2,np.arange(tspan_fcp-1)],0)*1 # check cluster is still existing except for the clusters at time tspan_fcp-1
    index = np.where(re == 1)[0]
    fcp_table[index_1[index],index] = 0
    if len(index)>0:
        if re[-1]==1:
            fcp_table[z_i[i,tspan_fcp-1],tspan_fcp-1] = 0
        
    z_i[i,:] = -1
    
    '''
    delete z_i
    fcp
    fcp_table
    '''

def p_ob(i,t,mc_num,g_ij,ob,fcp_table,b_mat_mu,q_arr,z_i,epsilon): 
    # probability for each cluster in backward forward algorithm with object i 
    
    prob = np.zeros((mc_num))
    
    index_c = np.where(fcp_table[:,2*t]!=0)[0] #existing cluster
    ob_i = np.concatenate((ob[i,:,t],ob[:,i,t]))
    index_d = np.where(ob_i>-1)[0]  #existing ob
    cur_index_mat = np.repeat(index_c[np.newaxis,:], len(index_d), 0) 
    # index_d ob row index_c cluster col
    cur_g_ij = np.concatenate((g_ij[i,:,t,:],g_ij[:,i,t,:]))
    s_g = np.copy(cur_g_ij[index_d,0]) #sender g
    r_g = np.copy(cur_g_ij[index_d,1]) #receiver g 
    b = b_mat_mu[s_g,r_g] # choose b according to sender g and receiver g 
    cur_q = np.copy(q_arr[s_g]) # choose q 
    s_g = np.repeat(s_g[:,np.newaxis], len(index_c), 1)
    r_g = np.repeat(r_g[:,np.newaxis], len(index_c), 1)
    b = np.repeat(b[:,np.newaxis], len(index_c), 1)
    cur_q = np.repeat(cur_q[:,np.newaxis], len(index_c), 1)
    z_i_double = np.concatenate((z_i[:,t*2],z_i[:,t*2]))
    cur_index_j = np.copy(z_i_double[index_d])
    cur_index_j = np.repeat(cur_index_j[:,np.newaxis], len(index_c), 1)
    
    ob_score = np.copy(ob_i[index_d])
    ob_score = np.repeat(ob_score[:,np.newaxis], len(index_c), 1)
    s1 = b*np.equal(cur_index_mat,cur_index_j)*np.equal(s_g,r_g)
    s2 = (cur_q+b)*np.not_equal(cur_index_mat,cur_index_j)*np.equal(s_g,r_g)
    s3 = b*np.equal(cur_index_mat,cur_index_j)*np.not_equal(s_g,r_g)
    s4 = epsilon*np.not_equal(cur_index_mat,cur_index_j)*np.not_equal(s_g,r_g)
    s = s1+s2+s3+s4
        
    cur_prob = 1/(np.exp(-s*ob_score)+np.exp(s-s*ob_score))
    cur_prob = np.sum(np.log(cur_prob),0)
    prob[index_c] =  cur_prob[:]+prob[index_c]
    
    s = np.amax(prob[index_c])
    prob[index_c] = np.abs(s)+prob[index_c]
    prob[index_c] = np.exp(prob[index_c])
    prob[index_c] = prob[index_c]*(10*100) #multiply exp(50) because prevent p_c * p_ob too small
    if np.sum(prob)==0:
        prob[index_c] = 1
#     print(prob[index_c],'prob')

    return prob

def s_g_ij(i,t,d_num,g_num,ob,g_mat,g_ij,g_prior,z_i,b_mat_mu,q_arr,epsilon): # sample g_ij
    for j in range(d_num):
        s = np.ones(g_num)
        if ob[i,j,t]>=0:
            s = s * c_p_g_ij(i, j, t, ob[i,j,t], 1,z_i,g_num,g_ij,b_mat_mu,q_arr,epsilon)
            a = g_ij[i,j,t,0]
            g_mat[i,a,t] = g_mat[i,a,t]-1
            s = (g_mat[i,:,t]+g_prior[t])*s
            s = npr.multinomial(1,s/np.sum(s))
            index = np.where(s==1)[0]
            g_ij[i,j,t,0] = np.copy(index)
            g_mat[i,index,t] = g_mat[i,index,t]+1
            
    for j in range(d_num):
        s = np.ones(g_num)
        if ob[j,i,t]>=0:
            s = s * c_p_g_ij(j, i, t, ob[j,i,t], 0,z_i,g_num,g_ij,b_mat_mu,q_arr,epsilon)
            b = g_ij[j,i,t,1]
            g_mat[i,b,t] = g_mat[i,b,t]-1
            s = (g_mat[i,:,t]+g_prior[t])*s
            s = npr.multinomial(1,s/np.sum(s))
            index = np.where(s==1)[0]
            g_ij[j,i,t,1] = np.copy(index)
            g_mat[i,index,t] = g_mat[i,index,t]+1
            
def c_p_g_ij(i,j,t,ob_score,order_ij,z_i,g_num,g_ij,b_mat_mu,q_arr,epsilon): 
# calculate probability for each g_ij for function s_g_ij
    
    c_i = np.copy(z_i[i,t*2])
    c_j = np.copy(z_i[j,t*2])
    g_i = np.arange(g_num)
    g_j = np.repeat(g_ij[i,j,t,1], g_num)
    if order_ij == 1:
        pass
    else:
        g_i = np.repeat(g_ij[i,j,t,0], g_num)
        g_j = np.arange(g_num)
        
    b = b_mat_mu[g_i,g_j]
    cur_q = np.copy(q_arr[g_j])
    s1 = b*np.equal(c_i,c_j)*np.equal(g_i,g_j)
    s2 = (cur_q+b)*np.not_equal(c_i,c_j)*np.equal(g_i,g_j)
    s3 = b*np.equal(c_i,c_j)*np.not_equal(g_i,g_j)
    s4 = epsilon*np.not_equal(c_i,c_j)*np.not_equal(g_i,g_j)
    s = s1+s2+s3+s4
    s = 1/(np.exp(-s*ob_score)+np.exp(s-s*ob_score))
    
    return s 

def s_blk(g_num,b_mu_ll,q_mu,b_v,q_v,  b_mat_mu,q_arr,b_mu_lk,b_mat_v,ob,g_ij,z_i): 
#sample b_lk q
    n_lk,n_lk1,m_l,m_l1 = get_nlk(ob,g_num,g_ij,z_i)
    for l in range(g_num):
        for k in range(l,g_num):
            samplenum = 100
            if l==k:
                b = np.zeros((samplenum,2))
                b[0,0] = b_mu_ll
                b[0,1] = q_mu
                mu = np.array([b_mu_ll,q_mu])
                var = np.array(([b_v,0],[0,q_v]))
                pg = PyPolyaGamma(seed=0)
                omegas = np.ones(2)
                x = np.array(([1,0],[1,1]))
                k_arr = np.array([n_lk1[l,l]-n_lk[l,l]/2,m_l1[l]-m_l[l]/2])

                for t in range(1,samplenum):
            
                    omegas[0] = pg.pgdraw(n_lk[l,l],b[t-1,0])
                    omegas[1] = pg.pgdraw(m_l[l],np.sum(b[t-1,:]))
                    omega = np.array(([omegas[0],0],[0,omegas[1]]))
                    v =  inv(np.dot(np.dot(np.transpose(x),omega),x)+inv(var))
                    m = np.dot(v,np.dot(np.transpose(x),np.transpose(k_arr))+np.dot(inv(var),mu))
                    s = npr.multivariate_normal(m, v)
                    b[t,0] = np.copy(s[0])
                    b[t,1] = np.copy(s[1])
                b_mat_mu[l,l] = np.sum(b[50:samplenum,0])/(samplenum-50)
                q_arr[l] = np.sum(b[50:samplenum,1])/(samplenum-50)

            else:
                b = np.zeros((samplenum,2))
                b[0,0] = b_mu_lk
                b[0,1] = b_mu_lk
                mu = np.array([b_mu_lk,b_mu_lk])
                var = np.copy(b_mat_v[:,:,l,k])
                pg = PyPolyaGamma(seed=0)
                omegas = np.ones(2)
                k_arr = np.array([n_lk1[l,k]-n_lk[l,k]/2,n_lk1[k,l]-n_lk[k,l]/2])
                x = np.array(([1,0],[0,1]))
                for t in range(1,samplenum):
                    omegas[0] = pg.pgdraw(n_lk[l,k],b[t-1,0])
                    omegas[1] = pg.pgdraw(n_lk[k,l],b[t-1,1])
                    omega = np.array(([omegas[0],0],[0,omegas[1]]))
                    
                    v =  inv(np.dot(np.dot(np.transpose(x),omega),x)+inv(var))
                    m = np.dot(v,np.dot(np.transpose(x),np.transpose(k_arr))+np.dot(inv(var),mu))
                    s = npr.multivariate_normal(m, v)
                    b[t,0] = np.copy(s[0])
                    b[t,1] = np.copy(s[1])
                b_mat_mu[l,k] = np.sum(b[50:samplenum,0])/(samplenum-50)
                b_mat_mu[k,l] = np.sum(b[50:samplenum,1])/(samplenum-50)

def get_nlk(ob,g_num,g_ij,z_i): #get n_lk m_l

    x,y,t = np.where(ob>-1)
    
    n_lk_mat = np.zeros((len(x),g_num,g_num))
    n_lk1_mat = np.zeros((len(x),g_num,g_num))
    m_l_mat = np.zeros((len(x),g_num))
    m_l1_mat = np.zeros((len(x),g_num))
    
    n_lk_mat[np.arange(len(x)),g_ij[x,y,t,0],g_ij[x,y,t,1]] = np.equal(z_i[x,t*2],z_i[y,t*2])*1
    n_lk1_mat[np.arange(len(x)),g_ij[x,y,t,0],g_ij[x,y,t,1]] = np.equal(z_i[x,t*2],z_i[y,t*2])*1*np.equal(ob[x,y,t],1)
    m_l_mat[np.arange(len(x)),g_ij[x,y,t,0]] = np.equal(g_ij[x,y,t,0],g_ij[x,y,t,1])*np.not_equal(z_i[x,t*2],z_i[y,t*2])*1
    m_l1_mat[np.arange(len(x)),g_ij[x,y,t,0]] = np.equal(g_ij[x,y,t,0],g_ij[x,y,t,1])*np.not_equal(z_i[x,t*2],z_i[y,t*2])*1*np.equal(ob[x,y,t],1)
    
    n_lk = np.sum(n_lk_mat,0)
    n_lk1 = np.sum(n_lk1_mat,0)
    m_l = np.sum(m_l_mat,0)
    m_l1 = np.sum(m_l1_mat,0)
    
    return n_lk,n_lk1,m_l,m_l1

def s_b_var(g_num,b_mat_mu,ivw_scale,b_mat_v,ivw_df,b_mu_lk): # sample b_lk variance
    for l in range(g_num):
        for k in range(l+1,g_num):
            x = np.array([b_mat_mu[l,k],b_mat_mu[k,l]])
            x = np.reshape(x, (1,2))
            var_x = np.dot(np.transpose(x)-b_mu_lk,x - b_mu_lk)
#             print(var_x,'var_x')
#             print(var_x,'var_x')
            if np.all(np.linalg.eigvals(ivw_scale + var_x) > 0)==0:
                print(np.linalg.eigvals(ivw_scale + var_x))
                
            b_mat_v[:,:,l,k] = invwishart.rvs(df = 1+ivw_df, scale = ivw_scale + var_x)
#             print(b_mat_v[:,:,l,k],'lk',l,k)
            
def predict(g_mat,g_prior,g_num,b_mat_mu,q_arr,epsilon,d_num,tspan,z_i):
#     for t in range(tspan):
#         print(g_mat[:,:,t],'g_mat',t)
#     print(g_prior,'g_prior')
    g = np.repeat(g_prior[np.newaxis,:], g_num, 0)
    g = np.repeat(g[np.newaxis,:,:], d_num, 0)
    s = np.sum(g_mat+g,1)
    s = np.repeat(s[:,np.newaxis,:],g_num,1)
    g = (g_mat+g)/s 
    b = 1/(1+np.exp(-b_mat_mu))
    s = np.diag(b_mat_mu)+q_arr

    b_q = 1/(1+np.exp(-s))
    b_q = np.diag(b_q)
    epsilon_mat = np.ones((g_num,g_num))/(1+np.exp(-epsilon))
    np.fill_diagonal(epsilon_mat, 0)
    re = np.zeros((d_num,d_num,tspan))

    for t in range(tspan):
        gg1 = np.dot(np.dot(g[:,:,t],b),np.transpose(g[:,:,t]))
        gg2 = np.dot(np.dot(g[:,:,t],b_q),np.transpose(g[:,:,t]))
        s = np.copy(z_i[:,t*2])
        z = np.repeat(s[:,np.newaxis],d_num,1)
        gg3 = np.dot(np.dot(g[:,:,t],epsilon_mat),np.transpose(g[:,:,t]))
        re[:,:,t] = np.equal(z,np.transpose(z))*1*gg1+np.not_equal(z,np.transpose(z))*1*gg2+\
        np.not_equal(z,np.transpose(z))*1*gg3

    return re

def f_oper(ts,mc_num,fcp,fcp_table,f_para): # fragmentation operation
    '''
    f_para:concentration parameter
    1 which row has cluster
    2 allow new cluster for each existing cluster
    3 mark new cluster
    '''
    prob_table = np.zeros((mc_num,mc_num))
    s1 = np.sum(fcp[:,:,ts],1) 

    index_c1 = np.where(fcp_table[:,ts]!=0)[0] # cluster exists for current timeslot
    index_c2 = np.where(fcp_table[:,ts+1]==0)[0] # cluster not exists for next timeslot
    fcp_table[index_c2[0:len(index_c1)],ts+1] = -1 # mark new cluster
    prob_table[index_c1, index_c2[0:len(index_c1)]] = f_para # set new cluster value at f-para 
    s = s1[index_c1]+f_para
    s = np.repeat(s[:,np.newaxis],mc_num , 1)
   
    prob_table[index_c1,:] = (fcp[index_c1,:,ts]+ prob_table[index_c1,:])/s
    
    return prob_table  

def f_init_oper(prob,fcp_table,z_i,f_para): # initial fragmentation operation

    index_c1 = np.where(fcp_table[:,0]==0)[0] # cluster not exists for current timeslot
#     print(index_c1,'index_c1')
    x,y = np.unique(z_i[:,0], return_counts=1)
    x[0] = np.copy(index_c1[0]) # because z_i[i,0]=-1
    y[0] = f_para
    fcp_table[index_c1[0],0] = -1
    prob[x,0] = y/np.sum(y)
    
def c_oper(ts,mc_num,fcp_table,fcp,c_para): # coagulation operation
    prob_table = np.zeros((mc_num,mc_num))

    index_c2 = np.where(fcp_table[:,ts+1]==0)[0] # cluster not exists for next timeslot
    index_new = np.where(fcp_table[:,ts]==-1)[0] # new cluster exists for current timeslot
    fcp_table[index_c2[0],ts+1] = -1 # mark new cluster
    x,y = np.where(fcp[:,:,ts]!=0) # find old cluster
    prob_table[x,y] = 1 # mark old cluster probability be 1
    index, counts = np.unique(y,return_counts=True) 
    s = np.sum(counts)
    for i in range(len(index)):
        prob_table[index_new,index[i]] = counts[i]/(s+c_para)
    
    prob_table[index_new,index_c2[0]] = c_para/(s+c_para)
    return prob_table

def normalize_p_c(p,t):
    s = np.amax(p[:,t])
    p[:,t] = p[:,t]/s
    
    
    
