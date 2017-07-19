
# coding: utf-8

# Modelo logístico-mutualista generalizado con mecanismo evolutivo (DA) y múltiples especies

# In[992]:

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as sc
import random


# In[993]:

#
# cantidad de especies (se deja como variable por si es posible extender el modelo a más especies)
# hay que considerar que por la dinámica de los traits, habría que hacer otros cambios
#
nesp = 2


# In[994]:

#
# Inicialización: se busca una solución que cumpla la condición dada para cambiar el parámetro "con"
# Esta condición no tiene que ser tal, pero puede ayudar cuando buscamos transiciones específicas
# para imponer, por ejemplo, que solo nos quedaremos con las soluciones donde haya un b concreto.
# Si la condición no se utiliza, lo único que hacemos es generar una secuencia de parámetros aleatorios
# cuya estructura es la siguiente: [[ro_1,ro_2,...,ro_nesp],[b_1,b_2,...,b_1nesp],[n_1,n_2,...,n_nesp],[w_1]]
# Los rangos de estas variables dependen de alpha, pero si consideramos un valor de alpha=0.5, que nos
# permite rangos simétricos de r y b, tenemos:
# ro entre [0,6.0], que dará r's entre [-1,1] y bo entre [0,0.3], que dará b's entre [-0.01,0.01]
# por su lado, aquí estamos escogiendo u1 y u2 entre [0,0.1] de modo que n1 y n2 salgan entre [0,50]
# puesto que tomamos nt=500 y queremos que n1 y n2 se tengan bastante margen para evolucionar
# por último estamos tomando a w entre [0,min(u1,u2)], ya que (w < u1) y (w < u2) son condiciones obligatorias.
# Finalmente, se crean todas listas vacías, se invoca la función update para rellenarlas con los valores iniciales e
# inicializamos todos los parámetros fijos; mu = 1.2/max(xoyo) para limitar las derivadas de los traits.
#
def ini(aa):
    global con,step,ulist,wlist,rlist,blist,xoyolist,m12list,m21list,ro,bo,u,w,a,c,l,nt,ee,mu,alpha
    con,step,ulist,wlist,rlist,blist,xoyolist,m12list,m21list = 0,0,[],[],[],[],[],[],[]
    indexlist = []
    while (con == 0):
        nt = 500
        ee = 0.001
        ro = np.random.uniform(0,6, size = 2).tolist()
        bo = np.random.uniform(0,0.03, size = 2).tolist()
        u = np.random.uniform(0,0.1, size = 2).tolist()
        w = np.random.uniform(0,min(u[0],u[1]), size = 1).tolist()
        #[ro,bo,u,w] = [[4.6464847612621725, 1.182792749728639], [0.01228447326865657, 0.009333730717957118], [0.07641603025496892, 0.08410035748197525], [0.042340725182942324]]
        a = [0.0001 for i in range(nesp)]
        c = [0.001 for i in range(nesp)]
        l = [1 for i in range(3)] #(np.random.uniform(1,2, size = nesp).tolist()
        alpha = aa
        bucle(u,w)
        if (step==1): #and np.any([np.any([np.real(b[i][j]) < 0 for j in range(nesp)]) for i in range(nesp)]):
            con = 1
    update(u,w,r,b,xoyo)
    mu = 1.2/max(xoyo).real


# In[995]:

#
# Actualización de las listas que monitoreamos: u y w (traits), r y b (parámetros) y xoyo (poblaciones)
# Añadimos también las listas de los traits cruzados m12 y m21, pues los graficaremos al final
#
def update(u,w,r,b,xoyo):
    global ulist,wlist,rlist,blist,xoyolist,m12list,m21list
    ulist.append(u)
    wlist.append(w)
    rlist.append(r)
    blist.append(b)
    m12list.append(alpha*(u[1]/ee + alpha*u[0]/ee)/(1 - alpha**2))
    m21list.append(alpha*(u[0]/ee + alpha*u[1]/ee)/(1 - alpha**2))
    xoyolist.append(xoyo)


# In[996]:

#
# Esta función calcula los puntos fijos, las poblaciones fijas y sus respectivos autovalores.
#
def ecuaciones(x):
    pops = poblaciones(x)
    return [(l[0]*mu*ee*pops[0]/(2*nt))*((f1u-f1w)*x[1] + abs(f1u-f1w)*(x[1] - 2*x[2]) + 
                                             f1u*(ee*nt - x[1]) + abs(f1u)*(ee*nt - 2*x[0] + 2*x[2] - x[1])),
        (l[1]*mu*ee*pops[1]/(2*nt))*((f2u-f2w)*x[0] + abs(f2u-f2w)*(x[0] - 2*x[2]) + 
                                     f2u*(ee*nt - x[0]) + abs(f2u)*(ee*nt - 2*x[1] + 2*x[2] - x[0])),
            (l[2]*mu*ee/(2*nt))*(pops[0]*((f1u-f1w)*x[1] + abs(f1u-f1w)*(x[1] - 2*x[2])) +
                                    pops[1]*((f2u-f2w)*x[0] + abs(f2u-f2w)*(x[0] - 2*x[2])))]
def poblaciones(x):
    ran = [ro[0]*((1 - alpha - alpha**2)*x[0] - (alpha**2)*x[1])/(1 - alpha**2),
         ro[1]*((1 - alpha - alpha**2)*x[1] - (alpha**2)*x[0])/(1 - alpha**2)]
    ban = [bo[0]*(((alpha**2)*x[0] + alpha*x[1])/(1 - alpha**2) - x[2]),
         bo[1]*(((alpha**2)*x[1] + alpha*x[0])/(1 - alpha**2) - x[2])]
    solusan = np.array([[(-a[0]*a[1] + ban[0]*ban[1] + ban[1]*c[1]*ran[0] - ban[0]*c[0]*ran[1] +
                pow((4*ban[1]*(ban[0]*c[0] + a[0]*c[1])*(a[1]*ran[0] + ban[0]*ran[1]) +
                (a[0]*a[1] - ban[1]*(ban[0] + c[1]*ran[0]) + ban[0]*c[0]*ran[1])**2),0.5)) / (2*ban[1]*(ban[0]*c[0] + a[0]*c[1])),
                          (-a[0]*a[1] + ban[0]*ban[1] - ban[1]*c[1]*ran[0] + ban[0]*c[0]*ran[1] +
                pow((4*ban[1]*(ban[0]*c[0] + a[0]*c[1])*(a[1]*ran[0] + ban[0]*ran[1]) +
                (a[0]*a[1] - ban[1]*(ban[0] + c[1]*ran[0]) + ban[0]*c[0]*ran[1])**2),0.5)) / (2*ban[0]*(ban[1]*c[1] + a[1]*c[0]))],
                          [(-a[0]*a[1] + ban[0]*ban[1] + ban[1]*c[1]*ran[0] - ban[0]*c[0]*ran[1] -
                pow((4*ban[1]*(ban[0]*c[0] + a[0]*c[1])*(a[1]*ran[0] + ban[0]*ran[1]) +
                (a[0]*a[1] - ban[1]*(ban[0] + c[1]*ran[0]) + ban[0]*c[0]*ran[1])**2),0.5)) / (2*ban[1]*(ban[0]*c[0] + a[0]*c[1])),
                          (-a[0]*a[1] + ban[0]*ban[1] - ban[1]*c[1]*ran[0] + ban[0]*c[0]*ran[1] -
                pow((4*ban[1]*(ban[0]*c[0] + a[0]*c[1])*(a[1]*ran[0] + ban[0]*ran[1]) +
                (a[0]*a[1] - ban[1]*(ban[0] + c[1]*ran[0]) + ban[0]*c[0]*ran[1])**2),0.5)) / (2*ban[0]*(ban[1]*c[1] + a[1]*c[0]))]])
    return solusan[index]
    
def puntosfijos(uu,ww):
    global fixedtraits,fixedpops
    fixedtraits = sc.root(ecuaciones,[float(np.real(uu[0])),float(np.real(uu[1])),float(np.real(ww[0]))]).x
    fixedpops = np.array(poblaciones(uu+ww)).astype(np.complex)


# In[997]:

#
# Esta función tiene un doble uso. Primero se utiliza en la inicialización, donde la condición con servirá para
# validar si es que se cumple una segunda condición que permita escoger c.i. que nos interesen especialmente.
# Segundo se utiliza en el cuerpo general del programa, donde ahí la condición step servirá para validar el avance
# de cada iteración. Esta función se invoca únicamente con los traits u y w, aunque devuelve varias listas más.
# Las soluciones las calculamos explícitamente a partir del resultado algebraico. En un sistema con más de dos poblaciones
# tendríamos que utilizar una aproximación numérica.
# El programa solo avanza siempre y cuando las nuevas soluciones sean: reales, estables y positivas.
#
def bucle(u,w):
    global r,b,equs,solus,jacs,lambdas,index,step,xoyo,totlambdas,totpops
    r = [ro[0]*((1 - alpha - alpha**2)*u[0] - (alpha**2)*u[1])/(1 - alpha**2),
         ro[1]*((1 - alpha - alpha**2)*u[1] - (alpha**2)*u[0])/(1 - alpha**2)]
    b = [bo[0]*(((alpha**2)*u[0] + alpha*u[1])/(1 - alpha**2) - w[0]),
         bo[1]*(((alpha**2)*u[1] + alpha*u[0])/(1 - alpha**2) - w[0])]
    solus = np.array([[(-a[0]*a[1] + b[0]*b[1] + b[1]*c[1]*r[0] - b[0]*c[0]*r[1] +
                pow((4*b[1]*(b[0]*c[0] + a[0]*c[1])*(a[1]*r[0] + b[0]*r[1]) +
                (a[0]*a[1] - b[1]*(b[0] + c[1]*r[0]) + b[0]*c[0]*r[1])**2),0.5)) / (2*b[1]*(b[0]*c[0] + a[0]*c[1])),
                          (-a[0]*a[1] + b[0]*b[1] - b[1]*c[1]*r[0] + b[0]*c[0]*r[1] +
                pow((4*b[1]*(b[0]*c[0] + a[0]*c[1])*(a[1]*r[0] + b[0]*r[1]) +
                (a[0]*a[1] - b[1]*(b[0] + c[1]*r[0]) + b[0]*c[0]*r[1])**2),0.5)) / (2*b[0]*(b[1]*c[1] + a[1]*c[0]))],
                          [(-a[0]*a[1] + b[0]*b[1] + b[1]*c[1]*r[0] - b[0]*c[0]*r[1] -
                pow((4*b[1]*(b[0]*c[0] + a[0]*c[1])*(a[1]*r[0] + b[0]*r[1]) +
                (a[0]*a[1] - b[1]*(b[0] + c[1]*r[0]) + b[0]*c[0]*r[1])**2),0.5)) / (2*b[1]*(b[0]*c[0] + a[0]*c[1])),
                          (-a[0]*a[1] + b[0]*b[1] - b[1]*c[1]*r[0] + b[0]*c[0]*r[1] -
                pow((4*b[1]*(b[0]*c[0] + a[0]*c[1])*(a[1]*r[0] + b[0]*r[1]) +
                (a[0]*a[1] - b[1]*(b[0] + c[1]*r[0]) + b[0]*c[0]*r[1])**2),0.5)) / (2*b[0]*(b[1]*c[1] + a[1]*c[0]))]]).astype(np.complex)
    #
    # la siguiente instrucción calcula los elementos del jacobiano de las soluciones; usamos un resultado analítico
    #
    jacs = [[[-solus[k][0]*(a[0] + c[0]*b[0]*solus[k][1]),solus[k][0]*b[0]*(1 - c[0]*solus[k][0])],
             [solus[k][1]*b[1]*(1 - c[1]*solus[k][1]),-solus[k][1]*(a[1] + c[1]*b[1]*solus[k][0])]] for k in range(len(solus))]
    lambdas = [np.linalg.eigvals(i) for i in np.array(jacs).astype(np.float64)] # aquí calculamos los autovalores directamente
    totlambdas = [-1 if np.all(np.sign(x.real) < 0) else 1 for x in lambdas] # etiquetamos los estables con -1 y al resto con +1
    scon = [x for x in range(len(totlambdas)) if totlambdas[x]==-1] # ubicamos las posiciones de los estables
    totpops = [1 if np.all(np.sign(x.real) > 0) else 0 for x in solus] # etiquetamos las poblaciones con parte real positiva con 1 y al resto con 0
    pcon = [x for x in range(len(totpops)) if totpops[x]==1] # ubicamos las poblaciones con parte real positiva
    if len(list(set(scon).intersection(pcon))) > 0: # si tenemos poblaciones estables con parte real positiva
        index = list(set(scon).intersection(pcon))[0] # escogemos la primera de la lista como solución
        step = 1 # y cambiamos el valor de step, lo que significa que el bucle puede dar un paso 
        xoyo = solus[index] # guardamos el valor de las poblaciones


# In[1002]:

#
# Este es el cuerpo principal del programa. Se está recurriendo nuevamente al bucle, utilizando la actualización de z.
# Nótese que el control se realiza con el parámetro step. Si luego del bucle su valor no cambia a 1, el programa detecta
# que se ha encontrado inestabilidad, ya sea porque no hay ninguna solución estable, o porque aquella no corresponde a 
# un conjunto de poblaciones reales y positivas. Si step cambia a 1, se procede a verificar que los parámetros r también
# sean reales y no excedan un límite preestablecido como 2. Finalmente se verifica que las poblaciones crezcan siempre
# o al menos no bajen de 1% del valor anterior. Luego de todo ello, se actualizan las listas.
#
def main(s,escala):
    global step,f1u,f1w,f2u,f2w,dfdu,dfdw,fin,u,w,uf,wf
    #
    # reseteamos fin a 0, puesto que si cambia a 1, el programa grafica cuando finaliza, después de un break
    #
    fin = 0
    for i in range(s):
        step = 0
        uf = ulist[-1]
        wf = wlist[-1]
        pops = xoyolist[-1]
        #
        # calculamos luego las derivadas parciales del r_eff respecto a cada trait propio
        #
        f1u = ro[0]*((1 - alpha - alpha**2) + bo[0]*(alpha**2)*pops[1]*(1 - c[0]*pops[0]))/(1 - alpha**2)
        f1w = bo[0]*pops[1]*(1 - c[0]*pops[0])
        f2u = ro[1]*((1 - alpha - alpha**2) + bo[1]*(alpha**2)*pops[0]*(1 - c[1]*pops[1]))/(1 - alpha**2)
        f2w = bo[1]*pops[0]*(1 - c[1]*pops[1])
        #
        # calculamos ahora las derivadas totales de los traits propios u1 y u2
        #
        dfdu = [(l[0]*mu*ee*pops[0]/(2*nt))*((f1u-f1w)*uf[1] + abs(f1u-f1w)*(uf[1] - 2*wf[0]) + 
                                             f1u*(ee*nt - uf[1]) + abs(f1u)*(ee*nt - 2*uf[0] + 2*wf[0] - uf[1])),
        (l[1]*mu*ee*pops[1]/(2*nt))*((f2u-f2w)*uf[0] + abs(f2u-f2w)*(uf[0] - 2*wf[0]) + 
                                     f2u*(ee*nt - uf[0]) + abs(f2u)*(ee*nt - 2*uf[1] + 2*wf[0] - uf[0]))]
        #
        # y la derivada del trait propio w
        #
        dfdw = (l[2]*mu*ee/(2*nt))*(pops[0]*((f1u-f1w)*uf[1] + abs(f1u-f1w)*(uf[1] - 2*wf[0])) +
                                    pops[1]*((f2u-f2w)*uf[0] + abs(f2u-f2w)*(uf[0] - 2*wf[0])))
        #
        # definimos la escala temporal y actualizamos los valores de los traits
        #
        dt = 10**(-escala) #+ max([divmod(np.log2(np.float64(abs(nu[i]/dfdn[i])))*1/np.log2(10),1)[0] for i in range(2)] + [divmod(np.log2(np.float64(abs(wu[0]/dfdw)))*1/np.log2(10),1)[0]]))
        u = [uf[i] + dt*dfdu[i] for i in range(nesp)]
        w = [wf[0] + dt*dfdw]
        #
        # condiciones de pare de traits: cuando la población cambia muy abruptamente, los límites de los traits podrían
        # terminar sobrepasando, a pesar de su limitación intrínseca por la construcción del sistema 
        #
        if (u[0]/ee > nt) or (u[1]/ee > nt) or (w[0]/ee > nt) or (u[0] < 0) or (u[1] < 0) or (w[0] < 0) or (w > u[0]) or (w > u[1]):
            f_out.write('limites de u/w por salto poblacional' + '\n')
            break
        #
        # ejecutamos el bucle central del programa
        #
        bucle(u,w)
        #
        # condiciones de pare de parámetros y/o poblaciones
        #
        # 1. si no se encuentra solución estable, o esta no corresponde a una población cuya parte real sea positiva
        #
        if (step == 0):
            f_out.write('inestabilidad' + '\n')
            break
        #    
        # 2. si las variaciones de los traits son muy pequeñas, asumimos que hemos llegado a la estabilidad
        #
        if (i>(iters//10)) and np.all([(np.real(u[j])/(ee*0.1)//1) == (np.real(ulist[-(iters//10)][j])/(ee*0.1)//1) for j in range(nesp)]) and np.all([(np.real(u[j])/ee//1) == (np.real(ulist[-(iters//20)][j])/ee//1) for j in range(nesp)]):
            f_out.write('ESS - ')
            fin = 1
            update(u,w,r,b,xoyo)
            break
        #    
        # 3. si alguno de los parámetros r o b se hace imaginario
        #
        if np.any([np.imag(r[i]) for i in range(nesp)]) or np.any([np.imag(b[i]) for i in range(nesp)]):
            f_out.write('r/b imaginarios' + '\n')
            #fin = 1
            break
        #    
        # 4. si la población que hemos seleccionado tiene parte imaginaria o su parte real es negativa
        #
        if np.any([np.imag(xoyo[i]) for i in range(nesp)]) or np.any([np.real(xoyo[i]) < 0 for i in range(nesp)]) or np.any([np.real(xoyo[i]) > 2*np.real(pops[i]) for i in range(nesp)]):
            f_out.write('poblacion negativa o imaginaria' + '\n')
            #fin = 1
            break
        #    
        # 5. si la población que hemos seleccionado está por debajo de 1 significa que va a converger hacia 0 asintóticamente
        #
        if np.any([np.real(xoyo[i]) < 1 for i in range(nesp)]):
            f_out.write('extincion' + '\n')
            #fin = 1
            break
        #    
        # si nada de lo anterior se cumple, actualizamos las listas y damos un pasito más
        #
        update(u,w,r,b,xoyo)


# In[999]:

#
# Esta función sirve para graficar los resultados del último bucle.
#
def plotnstats(i,aaa):
    path = 'graficos/'
    plt.clf()
    plt.plot(range(len(rlist)),[x[0] for x in rlist],'c--')
    plt.plot(range(len(rlist)),[x[1] for x in rlist],'b-.')
    plt.plot(range(len(blist)),[x[0]*100 for x in blist],'m-') # estoy multiplicando b12 y b21 por 100 para re-escalarlo
    plt.plot(range(len(blist)),[x[1]*100 for x in blist],'y:') # y que aparezcan en el mismo rango de r1 y r2
    plt.legend(['r1','r2','b12(x10^-2)','b21(x10^-2)'], loc='best') # esto facilita la visualización de dichas variables 
    plt.axhline(0, linewidth=0.75, color='black')
    plt.axis('tight')
    filename = 'alpha' + str(aaa) + 'ci_' + str(i) + '_parametros'
    plt.savefig(path + filename + '.png')
    plt.clf()
    plt.plot(range(len(ulist)),[x[0]/ee for x in ulist],'b-')
    plt.plot(range(len(ulist)),[x[1]/ee for x in ulist],'r-')
    plt.plot(range(len(wlist)),[x[0]/ee for x in wlist],'g-')
    plt.legend(['n1 ('+str(np.real(fixedtraits[0]//ee))[:-2]+')',
                'n2 ('+str(np.real(fixedtraits[1]//ee))[:-2]+')',
                'w ('+str(np.real(fixedtraits[2]//ee))[:-2]+')'], loc='best')
    plt.axhline(0, linewidth=0.75, color='black')
    plt.axhline(np.real(fixedtraits[0]/ee), linewidth=1, color='blue')
    plt.axhline(np.real(fixedtraits[1]/ee), linewidth=1, color='red')
    plt.axhline(np.real(fixedtraits[2]/ee), linewidth=1, color='green')
    plt.axis('tight')
    filename = 'alpha' + str(aaa) + 'ci_' + str(i) + '_traits propios'
    plt.savefig(path + filename + '.png')
    plt.clf()
    plt.plot(range(len(wlist)),[x[0]/ee for x in wlist],'g-')
    plt.plot(range(len(m12list)),m12list,'c--')
    plt.plot(range(len(m21list)),m21list,'m--')
    plt.legend(['w','m12','m21'], loc='best')
    plt.axhline(0, linewidth=0.75, color='black')
    plt.axis('tight')
    filename = 'alpha' + str(aaa) + 'ci_' + str(i) + '_traits cruzados'
    plt.savefig(path + filename + '.png')
    plt.clf()
    plt.plot(range(len(xoyolist)),[x[0] for x in xoyolist],'g.')
    plt.plot(range(len(xoyolist)),[x[1] for x in xoyolist],'r.')
    plt.legend(['Xo ('+str(np.real(fixedpops[0]//1))[:-2]+')',
                'Yo ('+str(np.real(fixedpops[1]//1))[:-2]+')'], loc='best')
    plt.axhline(0, linewidth=0.75, color='black')
    plt.axhline(np.real(fixedpops[0]), linewidth=1, color='green')
    plt.axhline(np.real(fixedpops[1]), linewidth=1, color='red')
    plt.axis('tight')
    filename = 'alpha' + str(aaa) + 'ci_' + str(i) + '_poblaciones'
    plt.savefig(path + filename + '.png')


# In[1003]:

#
# Esta es la función con la cual se generan condiciones iniciales aleatorias, se prueba su viabilidad y se calculan resultados
# de aquellas que producen poblaciones estables, estrictamente reales y positivas.
#
tries = 1   # intentos por alpha
iters = 10000000  # iteraciones por intento
alist = [0.1,0.2,0.3,0.4,0.5,0.6] # lista de alphas
for j in range(len(alist)):
    aaa =alist[j]
    print(aaa)
    for i in range(tries):
        f_out = open('graficos\prueba_' + str(j) +'ci'+ str(tries) + '_nt500_alpha' + str(aaa) +'_'+ str(iters)+'it.txt','a')
        f_out.write(str(i) + '\t')
        ini(aaa)
        f_out.write(str([ro,bo,u,w]) + '\t')
        main(iters, 0)
        if (len(xoyolist) >= iters or fin == 1):
            puntosfijos(ulist[-1],wlist[-1])
            if ((np.sign(blist[0][0]) > 0 and blist[-1][0] < -0.0001) or
                (np.sign(blist[0][1]) > 0 and blist[-1][1] < -0.0001)):
                plotnstats(i,aaa)
                f_out.write('transicion de mutualismo a parasitismo' + '\t') # definimos parasitismo si (b_ij < 0.0001)
            elif ((np.sign(blist[0][0]) > 0 and blist[-1][0] < 0 and blist[-1][0] > -0.0001) or
                (np.sign(blist[0][1]) > 0 and blist[-1][1] < 0 and blist[-1][1] > -0.0001)):
                plotnstats(i,aaa)
                f_out.write('transicion de mutualismo a comensalismo' + '\t') # definimos comensalismo si (0.0001 < b_ij < 0)
            elif (np.sign(rlist[0][0]) == np.sign(rlist[-1][0]) and np.sign(rlist[0][1]) == np.sign(rlist[-1][1]) and
                np.sign(blist[0][0]) == np.sign(blist[-1][0]) and np.sign(blist[0][1]) == np.sign(blist[-1][1])):
                plotnstats(i,aaa)
                f_out.write('sin transicion' + '\t')
            elif ((np.sign(blist[0][0]) < 0 and np.sign(blist[-1][0]) > 0) or
                  (np.sign(blist[0][1]) < 0 and np.sign(blist[-1][1]) > 0)):
                plotnstats(i,aaa)
                f_out.write('transicion de parasitismo a mutualismo' + '\t') # definimos mutualismo si (0 < b_ij)
            else:
                plotnstats(i,aaa)
                f_out.write('transicion entre mutualismos' + '\t')
            f_out.write(str([ro,bo,u,w]) + '\n')
        f_out.close()


# In[1001]:

f_out.close()


# In[ ]:



