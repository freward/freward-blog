<vue-mathjax></vue-mathjax>
# I - Reinforcement Learning - from Policy Gradient to Deep Deterministic Policy Gradient

L’apprentissage par renforcement est le domaine associé à l'enseignement de machine <span class = "tex2jax_ignore">(</span>agent<span class = "tex2jax_ignore">)</span> pour bien exécuter une tâche <span class="tex2jax_ignore">(</span>task<span class="tex2jax_ignore">)</span> en interagissant avec l'environnement <span class="tex2jax_ignore">(</span>environment<span class="tex2jax_ignore">)</span> et recevoir des récompenses <span class="tex2jax_ignore">(</span>reward<span class="tex2jax_ignore">)</span>.
Cette façon d’apprentissage est très similaire à la façon dont l’homme apprend en utilisant la mauvaise méthode de test. En prenant l'exemple d'un enfant en hiver, l'enfant aura tendance à se rapprocher du feu <span class = "tex2jax_ignore">(</span>car la récompense est chaleureuse<span class = "tex2jax_ignore">)</span>, mais également le feu est chaud, l'enfant aura tendance à éviter de toucher le feu <span class = "tex2jax_ignore">(</span>il le brûlera<span class = "tex2jax_ignore">)</span>.


Dans l'exemple ci-dessus, la récompense apparaît immédiatement, l'ajustement de l'action est
relativement facile. Cependant, dans des situations plus complexes où les récompenses sont lointains,
la situation devient plus compliquée. Comment obtenir la plus grande récompense accumulée tout au
long du processus? Les algorithmes d'apprentissage par renforcement <span class = "tex2jax_ignore">(</span>RL<span class = "tex2jax_ignore">)</span> visent à résoudre ce problème d'optimisation.

Voici les définitions des termes communs en RL:

* _Environment_ : est l’espace, le jeu, l’environnement avec lequel la machine interagit.
* _Agent_ : la machine observe l'environnement et génère une action en conséquence.
* _Policy_: la règle que l'agent suit pour obtenir le but.
* _Reward_: une récompense que l'agent a reçue de l'environnement pour avoir pris une action.
* _State_ : l'état de l'environnement observé par l'agent.
* _Episode_ : la séquence d'état et d'action jusqu'à la fin $s_1,a_1,s_2,a_2,...s_T, a_T$.
* _Accumulative Reward_ : la somme de toutes les récompenses reçues d'un épisode.


<div style="width:image width px; font-size:80%; text-align:center;">
<img src="https://i.imgur.com/nIUdsIm.jpg" align="center"/>
<div>Image 1: The interaction loop between agent and environment.</div>
</div>
</br>

Dans un état $s$, l'agent interagit avec l'environnement par l'action $a$,
conduisant à un nouvel état $ s_{t+1}$ et recevez une récompense $r_{t+1}$.
La boucle se répète ainsi jusqu'à ce que l'état final atteigne $s_T$.

Dans la section ci-dessous, je vais utiliser les termes Anglais à suivre au lieu de traduire en Français.

# 1 - Example
Cet exemple ci-dessous est à partir de openAI Gym, l'environnement nommé
[MountaincontinuousCar-v0](https://github.com/openai/gym/wiki/MountainCarContinuous-v0).

<div style="width:image width px; font-size:80%; text-align:center;">
    <img src="https://i.imgur.com/yGWmDei.jpg" alt="MountaincontinuousCar-v0" style="padding-bottom:0.5em;" />
    <div>Image 2: Un rendu de MountaincontinuousCar-v0.</div>
</div>
</br>

* _Goal_ : le but de ce jeu est de trouver une policy permettant de contrôler la voiture atteignant le drapeau.
* _Environment_ : des rampes et des voitures y circulent.
* _State_ : l'état du véhicule a 2 dimensions, les coordonnées du véhicule sur l'axe $x$ et la vitesse du véhicule au moment de la mesure.
* _Action_ : Force appliquée pour contrôler le véhicule, cependant, la force n'est pas assez forte pour pousser immédiatement la voiture au drapeau. La voiture devra faire des va-et-vient pour gagner suffisamment d’accélération et obtenir le drapeau.
* _Reward_ : À chaque pas que la voiture ne peut obtenir le drapeau, l'agent reçoit une reward $r=\frac{-a^2}{10}$,
et la reward 100 si elle atteint le cible. Alors si l'agent contrôle la voiture mais qu'il ne peut pas obtenir le drapeau, l'agent sera puni.
* _Terminal state_ : si l'agent obtenir le drapeau ou si le nombre d'étapes dépasse 998 étapes.

# 2 - Policy Gradient
Pour un exemple vivant, nous examinons un problème de jeu simple, le jeu Hare Egg.

<div style="width:image width px; font-size:80%; text-align:center;">
<img src="https://laptrinhcuocsong.com/images/game-hung-trung.png" align="center"/>
<div>Image 3: le jeu Hare Egg.</div>
</div>
</br>

Soit $\pi_\theta(a|s) = f(s, \theta)$ est le policy de l'agent, c'est une distribution de probabilité d'action $a$ à l'état $s$.

Dans le jeu Hare Egg, supposons que nous ayons 3 actions: aller à gauche, aller à droite ou rester immobile.
Correspondant à l'état à ce moment $s$ <span class = "tex2jax_ignore"> (</span>)la position du panier, la position de l'œuf tombant contre le panier,
la vitesse de chute des œufs...<span class = "tex2jax_ignore">) </span> nous aurons une distribution de probabilité d'action,
par exemple $ [0.1, 0.3, 0.5] $. La somme de toutes les probabilités d'action dans l'état $s$ est $1$, nous avons: $\sum_{a}\pi_\theta(a|s) = 1 $.
Soit $p(s_{t + 1} | a_t, s_t)$ la probabilité de distribution du prochain état lorsque l'agent est à l'état $s$ et exécute une action $a$.

Soit $\tau = s_1, a_1, s_2, a_2,..., s_T, a_T$ est la séquence de l'état $s_1$  à l'état $s_T$. La probabilité de $\tau$ est susceptible de se produire:

\[
\begin{eqnarray}
p_\theta(\tau) &=& p_\theta(s_1, a_1, s_2, a_2,...s_T, a_T) \\\\
               &=& p(s_1)\pi_\theta(a_1|s_1)p(s_2|s_1, a_1)\pi_\theta(a_2|s_2)...p(s_{T}|s_{T-1},a_{T-1})\pi_\theta(a_T|s_T) \\\\
               &=& p(s_1)\Pi_{t=1}^{t=T}\pi_\theta(a_t|s_t)p(s_{t+1}|s_t, a_t) \\\\
\end{eqnarray}
\]


Nous verrons que la distribution de probabilité de l'état $p(s_{t+1}|a_t, s_t)$ sera éliminée plus tard.

**L’apprentissage par renforcement vise à trouver des $\theta$ tels que:**

$$
\begin{eqnarray}
\theta^* &=& \arg\max_\theta E_{\tau\sim p_\theta(\tau)}\\big[r(\tau)\\big] \\\\
         &=& \arg\max_\theta E_{\tau\sim p_\theta(\tau)}\\bigg[\sum_t r(a_t, s_t)\\bigg]
\end{eqnarray}
$$

La formule montre que $\theta^\*$ est un ensemble de paramètres tels que l'attente de la reward accumulée de nombreux échantillons différents $\tau$, que nous collectons par la policy actuelle $\pi_\theta$ est la plus grande. <br />
Après $N$ épisodes différents, l'agent collecte $N$ échantillons différents $\tau$. La fonction objectif devient maintenant:

$$
\begin{eqnarray}
J(\theta) &=& E_{\tau\sim p_\theta(\tau)}\\bigg[\sum_t r(a_t, s_t)\\bigg] \\\\
          &=& \frac{1}{N} \sum_i\sum_t r(a_t, s_t)
\end{eqnarray}
$$

$J(\theta)$ est la moyenne des rewards accumulées des épisodes de $N$.<br/>
Nous pouvons également voir $J(\theta)$ sous la distribution de probabilité $p_\theta(\tau)$ comme ci-dessous:

$$
\begin{eqnarray}
J(\theta) &=& E_{\tau\sim p_\theta(\tau)}\\bigg[\sum_t r(a_t, s_t)\\bigg] \\\\
          &=& \int p_\theta(\tau) r(\tau) dr
\end{eqnarray}
$$

Continuant à examiner le gradient de la fonction objectif:

$$
\begin{eqnarray}
\nabla_\theta J(\theta) &=& E_{\tau\sim p_\theta(\tau)}\\bigg[\sum_t r(a_t, s_t)\\bigg] \\\\
&=& \int \nabla_\theta p_\theta(\tau) r(\tau) dr
\end{eqnarray}
$$
Mais nous avons aussi:
$$
\begin{eqnarray}
\nabla_\theta p_\theta(\tau) &=&  p_\theta(\tau) \frac{\nabla_\theta p_\theta(\tau)} {p_\theta(\tau)} \\\\
                             &=& p_\theta(\tau)\nabla_\theta \log p_\theta(\tau)
\end{eqnarray}
$$
** Notez ** que cette trick est utilisée souvant, donc:
$$
\begin{eqnarray}
\nabla_\theta J(\theta) &=& \int p_\theta(\tau) \nabla_\theta \log p_\theta(\tau) r(\tau) dr \\\\
                        &=& E_{\tau\sim p_\theta(\tau)}\\bigg[\nabla_\theta \log p_\theta(\tau) r(\tau)\\bigg]
\end{eqnarray}
$$

Examinons de plus sur $\log p_\theta(\tau)$, comme nous l'avons vu plus haut  $p_\theta(\tau) = p(s_1)\Pi_{t=1}^{t=T}\pi_\theta(a_t|s_t)p(s_{t+1}|s_t, a_t)$, nous avons:

$$
\begin{eqnarray}
\log p_\theta(\tau) = \log p(s_1) + \sum_{t=1}^{t=T}\log \pi_\theta(a_t|s_t) + \sum_{t=1}^{t=T}\log p(s_{t+1}|s_t, a_t)
\end{eqnarray}
$$

Finalement:
$$
\begin{eqnarray}
\nabla_\theta \log p_\theta(\tau) = \sum_{t=1}^{t=T}\nabla_\theta \log \pi_\theta(a_t|s_t)
\end{eqnarray}
$$

Ce résultat est intéressant parce que la dérivée par rapport à <span class="tex2jax_ignore">(</span>w.r.t<span class="tex2jax_ignore">)</span> $\theta$ de la fonction $\log p_\theta(\tau)$ ne dépend plus de la probabilité de transition de l’état $p(s_{t+1}|a_t, s_t)$, il ne dépend que de la distribution de probabilité de l’action $a_i$ exécutée par l’agent $s_i$.

Le gradient de la fonction objectif devient maintenant:
$$
\begin{eqnarray}
\nabla_\theta J(\theta) &=&  E_{\tau\sim p_\theta(\tau)}\\bigg[\nabla_\theta \log p_\theta(\tau) r(\tau)\\bigg] \\\\
&=& E_{\tau\sim p_\theta(\tau)}\\bigg[\sum_{t=1}^{t=T}\nabla_\theta \log\pi_\theta(a_t|s_t)\sum_{t=1}^{t=T} r(a_t, s_t)\\bigg]
\end{eqnarray}
$$

De même, après avoir passé $N$ épisodes , l'expectation de ce gradient est la suivante:
$$
\begin{eqnarray}
\nabla_\theta J(\theta) &=& \frac{1}{N}\sum_{i=1}^{N}\\bigg(\sum_{t=1}^{t=T}\nabla_\theta \log \pi_\theta(a_{i,t}|s_{i,t})\\bigg)\\bigg(\sum_{t=1}^{t=T} r(a_{i,t}, s_{i,t})\\bigg)
\end{eqnarray}
$$

Enfin, mettre à jour $\theta$ en utilisant un gradient ascendant:
$$
\begin{eqnarray}
\theta \leftarrow \theta + \nabla_\theta J(\theta)
\end{eqnarray}
$$

# 3 - L'algorithme REINFORCE
Résumez tous les résultats ci-dessus, nous avons l'algorithme REINFORCE comme ci-dessous:

1. Collecte $N$ samples {$\tau^i$} with the policy $\pi_\theta$
2. Calculate gradient: $\nabla_\theta J(\theta) = \frac{1}{N}\sum_{i=1}^{N}\\bigg(\sum_{t=1}^{t=T}\nabla_\theta \log \pi_\theta(a_{i,t}|s_{i,t})\\bigg)\\bigg(\sum_{t=1}^{t=T} r(a_{i,t}, s_{i,t})\\bigg) $
3. Update $\theta \leftarrow \theta + \nabla_\theta J(\theta)$

Maintenant, on se pauser pour regarder de plus sur le gradient de la fonction objectif. Écrivez sous une forme simple, nous avons:

$$
\begin{eqnarray}
\nabla_\theta J(\theta) = \frac{1}{N}\sum_{i=1}^{N}\nabla_\theta \log \pi_\theta(\tau_i)r(\tau_i)
\end{eqnarray}
$$
C'est exactement l'estimation du maximum de vraisemblance [EMV](https://fr.wikipedia.org/wiki/Maximum_de_vraisemblance) multipliée par la reward accumulée.
Optimiser la fonction objectif signifie également augmenter la probabilité de suivre les trajectoires $\tau$ qui donnent des reward cumulatives élevées.

# 4 - Quelques nouvelles définitions
$V^\pi(s)$: reward cumulative attendue à l'état $s$ par la policy $\pi$.<br/>
$Q^\pi(s,a)$: récompense cumulative attendue si exécuter l'action $a$ à l'état $s$ par la policy $\pi$.<br/>
La relation entre $V^\pi(s)$ et $Q^\pi(s,a)$: $V^\pi(s) = \sum_{a \in A}\pi_\theta(s,a)Q^\pi(s,a)$ - cela a du sens parce que $\pi_\theta(s,a)$ est la probabilité de faire une action $a$ à l’état $s$.<br/>
Nous avons également:
$$
\begin{eqnarray}
V^\pi(s_t) &=& E_\pi[G_t | S=s_t] \\\\
Q^\pi(s_t,a_t) &=& E_\pi[G_t|S=s_t, A=a_t]
\end{eqnarray}
$$
Dans lequel:<br/>
la somme de toutes les récompenses sera reçue de l'état $ s_t $ à l'avenir, avec le facteur de remise $ \ gamma $ appelé: 0 $ <\ gamma <1 $. Plus loin dans le futur, la récompense sera moins prise en compte, l'agent se soucie davantage des récompenses entrantes que des récompenses lointaines.

$G_t=\sum^{\infty}\_{k=0}\gamma^kR_{k+t+1}$: la somme de toutes les récompenses sera reçue de l'état $s_t$ à l'avenir, avec le facteur de remise $\gamma$: $0 < \gamma < 1$. Plus loin dans le futur, la reward sera moins prise en compte, l'agent concerne les rewards plus approches que des rewards lointaines.

## 4.1 - Les équations de Bellman

De la formule ci-dessus, nous avons:

$$
\begin{eqnarray}
V^\pi(s_t) &=& E_\pi\\bigg[G_t|S=s_t\\bigg] \\\\
           &=& E_\pi\\bigg[\sum^{\infty}\_{k=0}\gamma^kR_{k+t+1}|S=s_t\\bigg] \\\\
\end{eqnarray}
$$

Prenez la récompense $R_{t+1}$ reçue lorsque il passe de l'état $s_t$ à $s_{t+1}$ en dehors de $\sum$, nous obtenons:

$$
\begin{eqnarray}
E_\pi\\bigg[R_{t+1} + \gamma\sum^{\infty}\_{k=0}\gamma^kR_{k+t+2}|S=s_t\\bigg] &=& E_\pi[R_{t+1}|S=s_t] + \gamma E_\pi\\bigg[\sum^{\infty}\_{k=0}\gamma^kR_{k+t+2}|S=s_t\\bigg]
\end{eqnarray}
$$

Développez les 2 éxpectations de l'équation ci-dessus, nous avons:


$$
\begin{eqnarray}
E_\pi\\bigg[R_{t+1}|S=s_t\\bigg]=\sum_a \pi(s_t,a) \sum_{s_{t+1}} p(s_{t+1}|s_t, a)R(s_{t+1}|s_t, a)
\end{eqnarray}
$$
Mais:

$$
\begin{eqnarray}
\gamma E_\pi\\bigg[\sum^{\infty}\_{k=0}\gamma^kR_{k+t+2}|S=s_t\\bigg] = \sum_a \pi(s_t,a) \sum_{s_{t+1}} p(s_{t+1}|s_t, a)\gamma E_\pi\\bigg[\sum^\infty_{k=0} \gamma^k R_{t+k+2} | S = s_{t+1}\\bigg]
\end{eqnarray}
$$
Nous avons:
$$
\begin{eqnarray}
V^\pi(s_t) = \sum_a \pi(s_t,a) \sum_{s_{t+1}} p(s_{t+1}|s_t, a)\\Bigg[R(s_{t+1}|s_t, a) + \gamma E_\pi\bigg[\sum^\infty_{k=0} \gamma^k R_{t+k+2} | S = s_{t+1} \\bigg]\\Bigg]
\end{eqnarray}
$$
Remarquerez que:
$$
\begin{eqnarray}
E_\pi\\bigg[\sum^\infty_{k=0} \gamma^k R_{t+k+2} | S = s_{t+1}\\bigg] = V^\pi(s_{t+1})
\end{eqnarray}
$$

Enfin nous avons:
$$
\begin{eqnarray}
V^\pi(s_t) = \sum_a \pi(s_t,a) \sum_{s_{t+1}} p(s_{t+1}|s_t, a)\bigg[R(s_{t+1}|s_t, a) + \gamma  V^\pi(s_{t+1})\bigg]
\end{eqnarray}
$$

Faire la même chose avec $Q^\pi(s_t, a_t)$:
$$
\begin{eqnarray}
Q^\pi(s_t, a_t) = \sum_{s_{t+1}} p(s_{t+1}|s_t, a)\bigg[R(s_{t+1}|s_t, a) + \gamma \sum_{a_{t+1}} \pi(s_{t+1}, a_{t+1}) Q^\pi (s_{t+1}, a_{t+1}) \bigg]
\end{eqnarray}
$$
Combinez avec la relation entre $V^\pi$ and $Q^\pi$ ci-dessus, nous avons:
$$
\begin{eqnarray}
\sum_{a_{t+1}} \pi(s_{t+1}, a_{t+1}) Q^\pi (s_{t+1}, a_{t+1}) = V^\pi(s_{t+1})
\end{eqnarray}
$$
Donc:
$$
\begin{eqnarray}
Q^\pi(s_t, a_t) = \sum_{s_{t+1}} p(s_{t+1}|s_t, a_t)\\bigg[R(s_{t+1}|s_t, a_t) + \gamma  V^\pi(s_{t+1}) \\bigg]
\end{eqnarray}
$$

Par conséquent, si nous connaissons la valeur à l'état $ s_ {t + 1} $, nous pouvons facilement calculer la valeur à l'état $ s_t $. En résumé, nous avons 2 formules ci-dessous:
Tout ce qui précède montre que nous pouvons représenter la valeur de $Q^\pi$ et $V^\pi$ à l'état $s_t$ avec l'état $s_{t+1}$. Par conséquent, si nous connaissons la valeur à l'état $s_{t+1}$, nous pouvons facilement calculer la valeur à l'état $s_t$. En résumé, nous avons 2 formules ci-dessous:

$$
\begin{eqnarray}
V^\pi(s_t) &=& \sum_a \pi(s_t,a) \sum_{s_{t+1}} p(s_{t+1}|s_t, a)\\bigg[R(s_{t+1}|s_t, a) + \gamma  V^\pi(s_{t+1})\\bigg] \\\\
Q^\pi(s_t, a_t) &=& \sum_{s_{t+1}} p(s_{t+1}|s_t, a_t)\\bigg[R(s_{t+1}|s_t, a_t) + \gamma  V^\pi(s_{t+1}) \\bigg]
\end{eqnarray}
$$


Revenir au gradient de la fonction objectif, nous avons maintenant:
$$
\begin{eqnarray}
\nabla_\theta J(\theta) =  E_{\tau\sim p_\theta(\tau), a\sim\pi_\theta}\\bigg[\nabla_\theta \log\pi_\theta(a|s)Q^\pi(s,a)\\bigg]
\end{eqnarray}
$$

# 5 - Advantage
$$
\begin{eqnarray}
\nabla_\theta J(\theta) =  E_{\tau\sim p_\theta(\tau), a\sim\pi_\theta}\\bigg[\nabla_\theta \log\pi_\theta(a|s)Q^\pi(s,a)\\bigg]
\end{eqnarray}
$$

Le gradient de la fonction objectif indique que l'agent fera plus d'action $a$ s'il reçoit un $Q^\pi(s,a)$ élevé. En supposant que l'agent est à l'état $s$, le fait qu'il soit à l'état $s$ est déjà bon pour l'agent, l'exécution de toute action $a$ onnera un haut $Q^\pi(s,a)$ il ne peut donc pas discriminer ses actions $a$ et à partir de là, il ne sait pas quelle action $a$ est optimale. Par conséquent, nous avons besoin d’une base de référence pour comparer la valeur de $Q^\pi(s,a)$.<br/>

Comme dans la partie 4, nous avons $V^\pi(s)$ est l’attente d’une reward accumulée à l’état $s$, quelle que soit la décision prise par l’agent à l’état $s$, nous nous attendons à une reward accumulée $V^\pi(s)$ de là à la fin.
Par conséquent, une action $a_m$ est mauvaise si $Q^\pi(s,a_m)$ < $V^\pi(s)$ et une action $a_n$ est bonne si $Q^\pi(s,a_n)$ > $V^\pi(s)$. À partir de
là, nous avons un ligne de base pour comparer $Q^\pi(s,a)$ qui est $V^\pi(s)$. Le gradient de la fonction objectif peut maintenant être écrit:

$$
\begin{eqnarray}
\nabla_\theta J(\theta) =  E_{\tau\sim p_\theta(\tau), a\sim\pi_\theta}\\bigg[\nabla_\theta \log\pi_\theta(a|s)\\Big(Q^\pi(s,a)-V^\pi(s)\\Big)\\bigg]
\end{eqnarray}
$$

Si $Q^\pi(s,a)-V^\pi(s) < 0$,2 gradients ont des signes opposés, l'optimisation de la fonction objectif réduira la probabilité d'exécution de l'action  $a$ et $s$.<br/>
Nous appelons $A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)$ est l'avantage de l'action $a$ à l'état $s$.

# 6 - Stochastic Actor-Critic
Stochastic Actor signifie que la policy $\pi_\theta(a|s)$ est une distribution de probabilité d'actions à $s$. Nous appelons Stochastic Actor pour le distinguer de Deterministic Actor <span class="tex2jax_ignore">(</span>ou Deterministic Policy<span class="tex2jax_ignore">)</span> ce qui signifie que la politique n'est pas une distribution de probabilité d'actions à $s$, mais sous $s$ nous n'exécutons qu'une action déterministe. En d'autres termes, la probabilité d'exécution d'une action choisie $a=\mu_\theta(s)$ sur $s$ vaut 1 et toutes les autres actions valent 0.

Examinez le gradient de la fonction objectif que nous avons ci-dessus:
$$
\begin{eqnarray}
\nabla_\theta J(\theta) =  E_{\tau\sim p_\theta(\tau), a\sim\pi_\theta}\\bigg[\nabla_\theta \log\pi_\theta(a|s)\\Big(Q^\pi(s,a)-V^\pi(s)\\Big)\\bigg]
\end{eqnarray}
$$

À partir de l'équation de Bellman, nous avons la relation entre $Q^\pi$ et $V^\pi$, maintenant la fonction objectif devient:
$$
\begin{eqnarray}
\nabla_\theta J(\theta) =  E_{\tau\sim p_\theta(\tau), a\sim \pi_\theta}\\bigg[\nabla_\theta \log\pi_\theta(a|s)\\Big(R + \gamma V^\pi(s_{t+1})- V^\pi(s)\\Big)\\bigg]
\end{eqnarray}
$$

La fonction objectif dépend de 2 choses: la fonction policy et value $ V ^ \ pi $. En supposant que nous ayons une fonction d’approximation pour $ V ^ \ pi (s) $ est $ V_ \ phi (s) $ en fonction des paramètres $ \ phi $. <br/>
Nous appelons la fonction d'approbation de la politique $ \ pi_ \ theta $ Actor et la fonction d'appromixation de $ V_ \ phi $ est Critique.

La fonction objectif dépend de 2 choses: policy $\pi_\theta$ et la function value $V^\pi$. En supposant que nous ayons une fonction d’approximation pour $V^\pi(s)$ is $V_\phi(s)$ en fonction des paramètres $\phi$.<br/>
Nous appelons la fonction d'd’approximation de la policy $\pi_\theta$ est Actor  et la fonction d'appromixation de $V_\phi$ est Critic.

# 7 - L'algorithm Actor-Critic
À partir de l'algorithme REINFORCE, nous utilisons maintenant une fonction d'approximation supplémentaire pour la fonction de valeur $V_\phi$, qui change un peu et nous avons:

Batch Actor-Critic:<br/>
    1. Collecte une trajectoire $\tau$ à l'état terminal par la policy $\pi_\theta$<br/>
    2. Fit $V_\phi$ avec $y = \sum\_{i}^{T} r_i$<br/>
    3. Calcul $A(s_t,a_t) = r(s_t, a_t) + \gamma V_\phi(s_{t+1}) - V_\phi(s_{t})$<br/>
    4. Calcul $\nabla_\theta J(\theta) = \sum_i \nabla \log \pi_\theta (a_i|s_i) A^\pi (s_i, a_i)$<br/>
    5. Update $\theta \leftarrow \theta  + \alpha \nabla_\theta J(\theta)$<br/>
<br/>
<br/>
Ci-dessus, nous pouvons représenter $V_\phi(s) = r + V_\phi(s')$ selon l'équation de Bellman, Nous pouvons donc mettre à jour le modèle en ne sachant qu'une seule étape.<br/>
Online Actor-Critic:<br/>
    1. Avec policy $\pi_\theta$, exécutez 1 action $a \sim \pi_\theta(a|s)$ pour avoir $(s,a,s',r)$<br/>
    2. Fit $V_\phi (s)$ avec $r + V_\phi(s')$<br/>
    3. Trouvez $A(s_t,a_t) = r(s_t, a_t) + \gamma V_\phi(s_{t+1}) - V_\phi(s_{t})$<br/>
    4. Trouvez $\nabla_\theta J(\theta) = \sum_i \nabla \log \pi_\theta (a_i|s_i) A (s_i, a_i)$<br/>
    5. Mettre à jour $\theta \leftarrow \theta  + \alpha \nabla_\theta J(\theta)$<br/>
<br/>
<br/>
Alors, nous mettons à jour de manière itérative les deux fonctions d’approximation $V_\phi$ et $\pi_\theta$.

# 8 - De Stochastic Actor-Critic à Q-Learning
Examiner une policy comme suit:
$$
\begin{eqnarray}
\pi'(a_t|s_t) = 1 \ \text{if}\  a_t = \arg \max_{a_t} A^\pi(s_t, a_t)
\end{eqnarray}
$$
Policy $\pi'$ est une Deterministic Policy: étant donné une policy $\pi$ et en supposant que nous connaissons l'avantage des actions à l'état $s_t$ sous la policy $\pi$, nous choisissons toujours l'action avec le plus grand avantage à l'état $s$, la probabilité de cette action est 1, toutes les autres actions à $s_t$ sont 0.
La policy $\pi'$ sera toujours meilleure ou au moins égale à la policy $\pi$. Une policy est évaluée égale ou meilleure que les autres si:
$V^\pi(s) \leq V^{\pi'} (s) \forall s \in S$ : avec tout l'état $s$ dans l'espace d'état $S$, la valeur de retour $V^\pi(s)$ toujours inférieur ou égal à la valeur de retour $V^{\pi'} (s)$.<br/>
Par exemple, nous avons: à l'état $s$, nous avons 4 façons de passer à l'état $s'$ correspondant à 4 actions et avantages $A^\pi_1$, $A^\pi_2$, $A^\pi_3$, $A^\pi_4$.De l'état $s'$, nous continuons à suivre la policy $\pi$. De $s$ à $s'$, si nous choisissons de suivre la stochastic policy $\pi$,  l’avantage expecté est de $\sum_{a \in A} p(a)A^\pi_a$, cette quantité doit être inférieure à ou égal à $\max_a A^\pi_a$.

<div style="width:image width px; font-size:80%; text-align:center;">
<img src="https://i.imgur.com/yMtTahR.jpg" align="center"/>
<div>Image 4: Passage de l'état $s$ à $s'$.</div>
</div>
</br>
Therefore, with a policy $\pi$, we always can apply policy $\pi'$ over it to have a new policy equal or better.<br/>
We now have the algorithm as follow:<br/>
1. Evaluate $A^\pi(s,a)$ with different actions $a$ <br/>
2. Optimize $\pi \leftarrow \pi'$

But evaluating $A^\pi(s,a)$ is also equivalent to evaluate $Q^\pi(s,a)$ because $A^\pi(s,a) =  Q^\pi(s,a) - V^\pi(s) = r(s,a)  + \gamma V^\pi(s') - V^\pi(s)$, and the quantity $V^\pi(s)$ is the same for different actions $a$ at state $s$.<br/>
The algorithm now becomes:
1. Evaluate $Q^\pi(s,a) \leftarrow r(s,a)  + \gamma V^\pi(s') $ với các action $a$ khác nhau
2. Optimize $\pi \leftarrow \pi'$ : choose the action with highest $A$, it is also highest $Q$

Now we actually do not need to care about policy anymore, and the second step can be written as:
1. Evaluate $Q^\pi(s,a) \leftarrow r(s,a)  + \gamma V^\pi(s') $ for different actions $a$
2. $V^\pi(s) \leftarrow \max_a Q^\pi(s,a)$

If we use an approximation function for $V_\phi(s)$, we have the following algorithm:
1. Evaluate $V^\pi(s) \leftarrow \max_a \big(r(s,a)  + \gamma V^\pi(s')\big)$
2. $\phi \leftarrow \arg min_\phi \big(V^\pi(s) - V_\phi(s)\big)^2$

This algorithm is not good, in the first step we need to have reward $r(s, a)$ corresponding to different actions $a$, thus we need different simulations at a state $s$. To solve this, we could do the same analysis above with $Q(s,a)$ instead of $V(s)$.
1. Evaluate $y_i \leftarrow r(s,a_i)  + \gamma \max_{a'} Q_\phi(s', a') $
2. $\phi \leftarrow \arg min_\phi \\big(Q_\phi(s, a_i) - y_i\\big)^2$ $(\*)$

This is the Q-Learning algorithm. Notice that, reward $r$ above is not dependent on the state transition and on the policy $\pi$ used to generate the sample, therefore we only need the sample $(s, a, r, s')$ to improve the policy without knowing that sample is generated from which policy. Because of this reason, we call it off-policy. Later, we will have on-policy algorithms and they need the new samples generated by the current policy to improve itself <span class="tex2jax_ignore">(</span>cannot use experience from history<span class="tex2jax_ignore">)</span>.
We have the Online Q-Learning as follow:
1. Evaluate action $a$ to have $(s, a, s', r)$
2. Evaluate $y_i \leftarrow r(s,a_i)  + \gamma max_{a'} Q_\phi(s', a') $
3. $\phi \leftarrow \phi - \alpha \frac{dQ_\phi}{d\phi}(s,a) \big(Q_\phi(s, a_i) - y_i\big)$

Notice in the step 3, is it the gradient descent as same as where I marked $(\*)$ above? The answer is no, in fact, we ignore the derivative of $y_i$ by $\phi$ <span class="tex2jax_ignore">(</span>$y_i$ also depends on $\phi$<span class="tex2jax_ignore">)</span>. As a consequence, everytime we update $\phi$ with this algorithm, the value of the target $y_i$ is also changed! The target changes when we try to get closer, this makes the algorithm becomes unstable.<br/>
To solve this, we need another approximation function called target network, different from the train network we use to run. Target network will be update slowly and is used to evaluate $y$.<br/>
Another problem is that samples are generated continuously so they are correlated. The algorithm above is similar as Supervised Learning - we map a Q-value with a target, we want the samples are independent and identically distributed <span class="tex2jax_ignore">(</span>i.i.d<span class="tex2jax_ignore">)</span>. To break the correlation between samples, we could use an experience buffer: a list containing many samples from different episodes and we choose randomly a batch from the buffer and train our agent on that batch.

To sum up, for the algorithm to be stable, and possibly converge, we need:
- A separated target network called $Q_{\phi'}$.
- Experience buffer.


The algorithm now:
1. Execute action $a_i$ to have $(s_i, a_i, s_i', r_i)$ and put it into the buffer.
2. Sample randomly a batch $N$ samples from the buffer $(s_i, a_i, s_i', r_i)$.
3. Evaluate $y_i \leftarrow r(s,a_i)  + \gamma max_{a'} Q_{\phi'}(s', a')$ <span class="tex2jax_ignore">(</span>using target network here<span class="tex2jax_ignore">)</span>
4. $\phi \leftarrow \phi - \alpha \frac{1}{N}\sum_i^N \frac{dQ_\phi}{d\phi}(s,a_i) \big(Q_\phi(s, a_i) - y_i\big)$
5. Update target network $\phi' \leftarrow (1-\tau)\phi' + \tau \phi$ <span class="tex2jax_ignore">(</span>using $\tau \%$ of new train network to update target network<span class="tex2jax_ignore">)</span>
<br/>
<br/>
This algorithm is Deep Q-Network <span class="tex2jax_ignore">(</span>DQN<span class="tex2jax_ignore">)</span>.

# 9 - De Deep Q-Network à Deep Deterministic Policy Gradient

DQN algorithm is succeeded to approximate Q-value, but there is a limitation in step 3: we need to evaluate $Q_{\phi'}$ with all different actions to choose the highest $Q$. With discrete action space such as games, when the set of actions is just up down left right buttons, the number of actions is finite and small, this is possible. However, in continuous action space, for example with action in a range from 0 to 1 we need a different approach.
The first thing we could think of is to discretize the action space into bins, for example from 0 to 1, we could divide it by 5 or 10 bins. Another way is that we sample actions with uniform distribution on the action space and choose the highest $Q(s,a)$ at state $s$.

Deep Deterministic Policy Gradient <span class="tex2jax_ignore">(</span>DDPG<span class="tex2jax_ignore">)</span> has a nice approach, notice that:
$$
\begin{eqnarray}
max_{a} Q_{\phi}(s, a) = Q_\phi\big(s, \arg max_a Q_\phi(s,a)\big)
\end{eqnarray}
$$

Now, if we add an approximation function $\mu_\theta(s) = \arg max_a Q_\phi(s,a)$, we will have to find $\theta$ in which: $\theta \leftarrow  \arg max_\theta Q_\phi(s,\mu_\theta(s))$. This optimization evaluates the change of $Q_\phi$ w.r.t parameters $\theta$. We could evaluate this change by the chain rule as follow:
$\frac{dQ_\phi}{d\theta} = \frac{dQ_\phi}{d\mu} \frac{d\mu}{d\theta}$.

Notice that, $\mu_\theta(s)$ is a Deterministic Policy, therefore this method is called Deep Deterministic Policy Gradient.

The DDPG algorithm is as follow:

1. Execute action $a_i$ to have $(s_i, a_i, s_i', r_i)$ and put it into the buffer.
2. Sample a batch of $N$ samples from the buffer $(s_i, a_i, s_i', r_i)$.
3. Evaluate $y_i \leftarrow r(s,a_i)  + \gamma Q_{\phi'}\\big(s', \mu_{\theta'}(s')\\big)$ <span class="tex2jax_ignore">(</span>use both policy và Q target network here<span class="tex2jax_ignore">)</span>
4. $\phi \leftarrow \phi - \alpha \frac{1}{N}\sum_i^N \frac{dQ_\phi}{d\phi}(s,a_i) \big(Q_\phi(s, a_i) - y_i\big)$
4. $\theta \leftarrow \theta - \beta \frac{1}{N}\sum_i^N \frac{d\mu_\theta}{d\theta}(s) \frac{dQ_\phi}{da}(s, a_i)$
5. Update target network $\phi' \leftarrow (1-\tau)\phi' + \tau \phi$ và $\theta' \leftarrow (1-\tau)\theta' + \tau \theta$
<br/>
<br/>
Notice in the  DDPG implementation:
- Because actions in DDPG are always deterministic, thus to explore the environment <span class="tex2jax_ignore">(</span>we do not want the agent always exploit the best trajectory in its knowledge, there may be a better path out there<span class="tex2jax_ignore">)</span>, a small action noise will be added into the action from the agent.
The noise in the [original paper](https://arxiv.org/abs/1509.02971) is a stochastic process named Ornstein–Uhlenbeck process <span class="tex2jax_ignore">(</span>OU process<span class="tex2jax_ignore">(</span>. The authors choosed this process because the experiments gave good result, however other experiments conducted by different groups showing that other noise such as Gaussian Noise give the same performance.
- Implementation of action noise in the library [keras-rl](https://github.com/keras-rl/keras-rl) is OU process, however when I run, this noise is not decayed w.r.t time. Hoever, we need a big noise at the beginning <span class="tex2jax_ignore">(</span>to explore the environment<span class="tex2jax_ignore">)</span> and decrease after many episodes. This can be done if before we add the noise to the action, we multiply it with a quantity epsilon and the epsilon decays to 0 w.r.t time.
- Besides adding noise into the action before executing it on the environment, they also add Gaussian Noise on the node of Neural Network. Reference paper [here](https://openai.com/blog/better-exploration-with-parameter-noise/). You could find the implementation o actor network noise in this library [stable baselines](https://stable-baselines.readthedocs.io/en/master/modules/ddpg.html#action-and-parameters-noise).

# 10 - Conclusion

To sum up, we went throught the Policy Gradient algorithm to DQN and DDPG.

Take away:
- Policy Gradient is an on-policy and a stochastic policy.
- Q-learning, DQN, DDPG are off-policy and deterministic policies.
- Always have a deterministic policy better than a stochastic policy.


