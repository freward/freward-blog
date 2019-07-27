<vue-mathjax></vue-mathjax>
# I - Reinforcement Learning - from Policy Gradient to Deep Deterministic Policy Gradient
Reinforcement Learning is the field related to teaching machine <span class="tex2jax_ignore">(</span>agent<span class="tex2jax_ignore">)</span> to perform well a task by interacting with the environment and receive rewards.
This way of learning is very similar to the way human learns from the environment by the wrong test method. Taking an example of a child in the winter, the child will tend to get closer to the fire <span class="tex2jax_ignore">(</span>because the reward is warm<span class="tex2jax_ignore">)</span>, but touching the fire is hot, the child will tend to avoid touching the fire <span class="tex2jax_ignore">(</span>it will burn him<span class="tex2jax_ignore">)</span>.

In the above example, the reward appears immediately, the action adjustment is
relatively easy. However, in more complex situations where rewards are far in the future, this becomes more complicated. How to achieve highest accumulative reward throughout the process? Reinforcement Learning algorithms <span class = "tex2jax_ignore">(</span> RL<span class = "tex2jax_ignore">)</span> are to solve this optimization problem.

Here are the definitions of common terms in RL:

* _Environment_ : is the space, the game, the environment that the machine interacts with.
* _Agent_ : the machine observes the environment and generates action accordingly.
* _Policy_: the rule which the agent follows to get the goal.
* _Reward_: a reward that the agent received from the environment for taking an action.
* _State_ : the state of the environment that the agent observes.
* _Episode_ : the sequence of state and action until it finishes $s_1,a_1,s_2,a_2,...s_T, a_T$
* _Accumulative Reward_ : the sum of all rewards received from an episode.


<div style="width:image width px; font-size:80%; text-align:center;">
<img src="https://i.imgur.com/nIUdsIm.jpg" align="center"/>
<div>Image 1: The interaction loop between agent and environment.</div>
</div>
</br>

At a state $s$, agent interact with environment by action $a$,
leading to a new state $s_{t+1}$ and receive a reward $r_{t+1}$.
The loop repeats like this until the final state reached $s_T$.

# 1 - Example
This example below is from openAI Gym, environment named
[MountaincontinuousCar-v0](https://github.com/openai/gym/wiki/MountainCarContinuous-v0).

<div style="width:image width px; font-size:80%; text-align:center;">
    <img src="https://i.imgur.com/yGWmDei.jpg" alt="MountaincontinuousCar-v0" style="padding-bottom:0.5em;" />
    <div>Image 2: A render from MountaincontinuousCar-v0.</div>
</div>
</br>

* _Goal_ : the goal of this game is to find a policy to control the car reaching the flag.
* _Environment_ : ramps and cars running in it.
* _State_ : the state of the vehicle has 2 dimensions, the coordinates of the vehicle in the $x$ axis and the speed of the vehicle at the time of measurement.
* _Action_ : Force applied to control the vehicle, however the force is not strong enough in order to push the car at once to the flag. The car will need to go back and forth on the sides of gain enough acceleration and reach the flag.
* _Reward_ : With every step that the car cannot reach the flag, the agent gets a reward $r=\frac{-a^2}{10}$,
and a reward 100 if it reach the target. Thus, if the agent controls the car but it cannot reach the flag, the agent will be punished.
* _Terminal state_ : ff the agent reaches the flag or the step number exceeds 998 steps.

# 2 - Policy Gradient
For a lively example, we examine a simple game problem, Hare Egg game.

<div style="width:image width px; font-size:80%; text-align:center;">
<img src="https://laptrinhcuocsong.com/images/game-hung-trung.png" align="center"/>
<div>Image 3: Hare Egg game.</div>
</div>
</br>

Let $\pi_\theta(a|s) = f(s, \theta)$ is the policy of agent, it is a probability distribution of action $a$ at state $s$.

In Hare Egg game, assume that we have 3 actions: go left, go right or stand still.
Corresponding to the current state $s$ <span class="tex2jax_ignore">(</span>the position of the basket, the position of the egg falling against the basket,
speed of eggs falling...<span class="tex2jax_ignore">)</span> we will have a probability distribution of action,
for example $[0.1, 0.3, 0.5]$. The sum of all action probability at state $s$ is $1$, we have: $\sum_{a}\pi_\theta(a|s) = 1$.
Let $p(s_{t+1}|a_t, s_t)$ is the probability distribution of the next state when the agent is at state $s$ and executes an action $a$.

Let $\tau = s_1, a_1, s_2, a_2,..., s_T, a_T$ is the sequence from state $s_1$ to state $s_T$. The probability of $\tau$ is likely to happen:

\[
\begin{eqnarray}
p_\theta(\tau) &=& p_\theta(s_1, a_1, s_2, a_2,...s_T, a_T) \\\\
               &=& p(s_1)\pi_\theta(a_1|s_1)p(s_2|s_1, a_1)\pi_\theta(a_2|s_2)...p(s_{T}|s_{T-1},a_{T-1})\pi_\theta(a_T|s_T) \\\\
               &=& p(s_1)\Pi_{t=1}^{t=T}\pi_\theta(a_t|s_t)p(s_{t+1}|s_t, a_t) \\\\
\end{eqnarray}
\]



We will see that the probability distribution of state $p(s_{t+1}|a_t, s_t)$ will be eliminated later.

**The goal of reinforcement learning is to find $\theta$ such that:**

$$
\begin{eqnarray}
\theta^* &=& \arg\max_\theta E_{\tau\sim p_\theta(\tau)}\\big[r(\tau)\\big] \\\\
         &=& \arg\max_\theta E_{\tau\sim p_\theta(\tau)}\\bigg[\sum_t r(a_t, s_t)\\bigg]
\end{eqnarray}
$$

From the formula we can see $\theta^\*$ is a set of parameters such that the expectation of accumulative reward from many different samples $\tau$, which we collect by the current policy $\pi_\theta$ is biggest.<br />
After $N$ different episodes, the agent collects $N$ different samples $\tau$. The objective function now becomes:

$$
\begin{eqnarray}
J(\theta) &=& E_{\tau\sim p_\theta(\tau)}\\bigg[\sum_t r(a_t, s_t)\\bigg] \\\\
          &=& \frac{1}{N} \sum_i\sum_t r(a_t, s_t)
\end{eqnarray}
$$

$J(\theta)$ is the average of accumulative reward from $N$ episodes.<br/>
We can also see $J(\theta)$ under the probability distribution $p_\theta(\tau)$ as below:

$$
\begin{eqnarray}
J(\theta) &=& E_{\tau\sim p_\theta(\tau)}\\bigg[\sum_t r(a_t, s_t)\\bigg] \\\\
          &=& \int p_\theta(\tau) r(\tau) dr
\end{eqnarray}
$$

Continuing to examine gradient of the objective function:

$$
\begin{eqnarray}
\nabla_\theta J(\theta) &=& E_{\tau\sim p_\theta(\tau)}\\bigg[\sum_t r(a_t, s_t)\\bigg] \\\\
&=& \int \nabla_\theta p_\theta(\tau) r(\tau) dr
\end{eqnarray}
$$
But we also have:
$$
\begin{eqnarray}
\nabla_\theta p_\theta(\tau) &=&  p_\theta(\tau) \frac{\nabla_\theta p_\theta(\tau)} {p_\theta(\tau)} \\\\
                             &=& p_\theta(\tau)\nabla_\theta \log p_\theta(\tau)
\end{eqnarray}
$$
**Note** that this trick is usually used, thus:
$$
\begin{eqnarray}
\nabla_\theta J(\theta) &=& \int p_\theta(\tau) \nabla_\theta \log p_\theta(\tau) r(\tau) dr \\\\
                        &=& E_{\tau\sim p_\theta(\tau)}\\bigg[\nabla_\theta \log p_\theta(\tau) r(\tau)\\bigg]
\end{eqnarray}
$$

Take a closer look at $\log p_\theta(\tau)$, as we have seen above $p_\theta(\tau) = p(s_1)\Pi_{t=1}^{t=T}\pi_\theta(a_t|s_t)p(s_{t+1}|s_t, a_t)$, we have:
$$
\begin{eqnarray}
\log p_\theta(\tau) = \log p(s_1) + \sum_{t=1}^{t=T}\log \pi_\theta(a_t|s_t) + \sum_{t=1}^{t=T}\log p(s_{t+1}|s_t, a_t)
\end{eqnarray}
$$

Finally:
$$
\begin{eqnarray}
\nabla_\theta \log p_\theta(\tau) = \sum_{t=1}^{t=T}\nabla_\theta \log \pi_\theta(a_t|s_t)
\end{eqnarray}
$$
This result is beautiful because the derivative with respect to <span class="tex2jax_ignore">(</span>w.r.t<span class="tex2jax_ignore">)</span> $\theta$ of the function $\log p_\theta(\tau)$ is no longer dependent on the transition probability of state $p(s_{t+1}|a_t, s_t)$, it is only dependent on the probability distribution of action $a_i$ which the agent execute on $s_i$.

Gradient of the objective function now becomes:
$$
\begin{eqnarray}
\nabla_\theta J(\theta) &=&  E_{\tau\sim p_\theta(\tau)}\\bigg[\nabla_\theta \log p_\theta(\tau) r(\tau)\\bigg] \\\\
&=& E_{\tau\sim p_\theta(\tau)}\\bigg[\sum_{t=1}^{t=T}\nabla_\theta \log\pi_\theta(a_t|s_t)\sum_{t=1}^{t=T} r(a_t, s_t)\\bigg]
\end{eqnarray}
$$

Similarly, after experiencing $N$ episodes, the expectation of this gradient is:
$$
\begin{eqnarray}
\nabla_\theta J(\theta) &=& \frac{1}{N}\sum_{i=1}^{N}\\bigg(\sum_{t=1}^{t=T}\nabla_\theta \log \pi_\theta(a_{i,t}|s_{i,t})\\bigg)\\bigg(\sum_{t=1}^{t=T} r(a_{i,t}, s_{i,t})\\bigg)
\end{eqnarray}
$$

Finally, we update $\theta$ using gradient ascent:
$$
\begin{eqnarray}
\theta \leftarrow \theta + \nabla_\theta J(\theta)
\end{eqnarray}
$$

# 3 - REINFORCE algorithm
Sum up all the result above, we have the REINFORCE algorithm as below:

1. Collect $N$ samples {$\tau^i$} with the policy $\pi_\theta$
2. Calculate gradient: $\nabla_\theta J(\theta) = \frac{1}{N}\sum_{i=1}^{N}\\bigg(\sum_{t=1}^{t=T}\nabla_\theta \log \pi_\theta(a_{i,t}|s_{i,t})\\bigg)\\bigg(\sum_{t=1}^{t=T} r(a_{i,t}, s_{i,t})\\bigg) $
3. Update $\theta \leftarrow \theta + \nabla_\theta J(\theta)$

Now, let's stop to look closer on the gradient of the objective function. Write in a simple form, we have:

$$
\begin{eqnarray}
\nabla_\theta J(\theta) = \frac{1}{N}\sum_{i=1}^{N}\nabla_\theta \log \pi_\theta(\tau_i)r(\tau_i)
\end{eqnarray}
$$
This is exactly the maximum likelihood estimation [MLE](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation) multiply with the accumulative reward.
Optimizing the objective function also means increasing the probability to follow sequences $\tau$ which give high accumulative rewards.

# 4 - Some new definitions
$V^\pi(s)$: expected accumulative reward at state $s$ by policy $\pi$.<br/>
$Q^\pi(s,a)$: expected accumulative reward if execute action $a$ at state $s$ by policy $\pi$.<br/>
The relationship between $V^\pi(s)$ and $Q^\pi(s,a)$: $V^\pi(s) = \sum_{a \in A}\pi_\theta(s,a)Q^\pi(s,a)$ - this makes sense because $\pi_\theta(s,a)$ is the probability of doing action $a$ at state $s$.<br/>
TWe also have:
$$
\begin{eqnarray}
V^\pi(s_t) &=& E_\pi[G_t | S=s_t] \\\\
Q^\pi(s_t,a_t) &=& E_\pi[G_t|S=s_t, A=a_t]
\end{eqnarray}
$$
In which:<br/>
$G_t=\sum^{\infty}\_{k=0}\gamma^kR_{k+t+1}$: sum of all reward will be received from state $s_t$ to the future, with $\gamma$ called discount factor: $0 < \gamma < 1$. The farther into the future, the reward will be discounted more, agent cares more about incoming rewards than far future rewards.

## 4.1 - Bellman Equations

From the formula above, we have:

$$
\begin{eqnarray}
V^\pi(s_t) &=& E_\pi\\bigg[G_t|S=s_t\\bigg] \\\\
           &=& E_\pi\\bigg[\sum^{\infty}\_{k=0}\gamma^kR_{k+t+1}|S=s_t\\bigg] \\\\
\end{eqnarray}
$$

Take the reward $R_{t+1}$ received when going from state $s_t$ to $s_{t+1}$ to outside the $\sum$, we get:

$$
\begin{eqnarray}
E_\pi\\bigg[R_{t+1} + \gamma\sum^{\infty}\_{k=0}\gamma^kR_{k+t+2}|S=s_t\\bigg] &=& E_\pi[R_{t+1}|S=s_t] + \gamma E_\pi\\bigg[\sum^{\infty}\_{k=0}\gamma^kR_{k+t+2}|S=s_t\\bigg]
\end{eqnarray}
$$

Expand 2 expected values of the equation above, we have:

$$
\begin{eqnarray}
E_\pi\\bigg[R_{t+1}|S=s_t\\bigg]=\sum_a \pi(s_t,a) \sum_{s_{t+1}} p(s_{t+1}|s_t, a)R(s_{t+1}|s_t, a)
\end{eqnarray}
$$
But:

$$
\begin{eqnarray}
\gamma E_\pi\\bigg[\sum^{\infty}\_{k=0}\gamma^kR_{k+t+2}|S=s_t\\bigg] = \sum_a \pi(s_t,a) \sum_{s_{t+1}} p(s_{t+1}|s_t, a)\gamma E_\pi\\bigg[\sum^\infty_{k=0} \gamma^k R_{t+k+2} | S = s_{t+1}\\bigg]
\end{eqnarray}
$$
We have:
$$
\begin{eqnarray}
V^\pi(s_t) = \sum_a \pi(s_t,a) \sum_{s_{t+1}} p(s_{t+1}|s_t, a)\\Bigg[R(s_{t+1}|s_t, a) + \gamma E_\pi\bigg[\sum^\infty_{k=0} \gamma^k R_{t+k+2} | S = s_{t+1} \\bigg]\\Bigg]
\end{eqnarray}
$$
Notice that:
$$
\begin{eqnarray}
E_\pi\\bigg[\sum^\infty_{k=0} \gamma^k R_{t+k+2} | S = s_{t+1}\\bigg] = V^\pi(s_{t+1})
\end{eqnarray}
$$

Finally we have:
$$
\begin{eqnarray}
V^\pi(s_t) = \sum_a \pi(s_t,a) \sum_{s_{t+1}} p(s_{t+1}|s_t, a)\bigg[R(s_{t+1}|s_t, a) + \gamma  V^\pi(s_{t+1})\bigg]
\end{eqnarray}
$$
Doing similar thing with $Q^\pi(s_t, a_t)$:
$$
\begin{eqnarray}
Q^\pi(s_t, a_t) = \sum_{s_{t+1}} p(s_{t+1}|s_t, a)\bigg[R(s_{t+1}|s_t, a) + \gamma \sum_{a_{t+1}} \pi(s_{t+1}, a_{t+1}) Q^\pi (s_{t+1}, a_{t+1}) \bigg]
\end{eqnarray}
$$
Combine with the relation between $V^\pi$ and $Q^\pi$ above, we have:
$$
\begin{eqnarray}
\sum_{a_{t+1}} \pi(s_{t+1}, a_{t+1}) Q^\pi (s_{t+1}, a_{t+1}) = V^\pi(s_{t+1})
\end{eqnarray}
$$
Thus:
$$
\begin{eqnarray}
Q^\pi(s_t, a_t) = \sum_{s_{t+1}} p(s_{t+1}|s_t, a_t)\\bigg[R(s_{t+1}|s_t, a_t) + \gamma  V^\pi(s_{t+1}) \\bigg]
\end{eqnarray}
$$

All of the above show that we can represent the value of $Q^\pi$ and $V^\pi$ at state $s_t$ with state $s_{t+1}$. Therefore, if we know the value at state $s_{t+1}$, we can easily calculate the value at state $s_t$. To sum up, we have 2 formulas below:
$$
\begin{eqnarray}
V^\pi(s_t) &=& \sum_a \pi(s_t,a) \sum_{s_{t+1}} p(s_{t+1}|s_t, a)\\bigg[R(s_{t+1}|s_t, a) + \gamma  V^\pi(s_{t+1})\\bigg] \\\\
Q^\pi(s_t, a_t) &=& \sum_{s_{t+1}} p(s_{t+1}|s_t, a_t)\\bigg[R(s_{t+1}|s_t, a_t) + \gamma  V^\pi(s_{t+1}) \\bigg]
\end{eqnarray}
$$


Going back with the gradient of the objective function, now we have:
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
Gradient of the objective function shows that the agent will do more action $a$ if it receives a high $Q^\pi(s,a)$. Assuming that the agent is at state $s$, the fact that it is at state $s$ is already good for the agent, executing any action $a$ will give back a high $Q^\pi(s,a)$ so it cannot discriminate its actions $a$ and from there, it does not know which action  $a$ is optimal. Therefore, we need a baseline to compare the value of $Q^\pi(s,a)$.<br/>
As in the part 4, we have $V^\pi(s)$ is the expectation of accumulative reward at state $s$, no matter what action the agent will take at state $s$, we expect an accumulative reward $V^\pi(s)$ from there to the end.
Therefore, an action $a_m$ is bad if $Q^\pi(s,a_m)$ < $V^\pi(s)$ and an action $a_n$ is good if $Q^\pi(s,a_n)$ > $V^\pi(s)$. From here we have 1 baseline to compare $Q^\pi(s,a)$ which is $V^\pi(s)$. The gradient of objective function now can be written:
$$
\begin{eqnarray}
\nabla_\theta J(\theta) =  E_{\tau\sim p_\theta(\tau), a\sim\pi_\theta}\\bigg[\nabla_\theta \log\pi_\theta(a|s)\\Big(Q^\pi(s,a)-V^\pi(s)\\Big)\\bigg]
\end{eqnarray}
$$

If $Q^\pi(s,a)-V^\pi(s) < 0$, 2 gradients have opposite signs, optimizing the objective function will decrease the probability of executing action $a$ at $s$.<br/>
We call $A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)$ is the Advantage of action $a$ at state $s$.

# 6 - Stochastic Actor-Critic
Stochastic Actor means the policy $\pi_\theta(a|s)$ is a probability distribution of actions at $s$. We call Stochastic Actor to distinguish it from Deterministic Actor <span class="tex2jax_ignore">(</span>or Deterministic Policy<span class="tex2jax_ignore">)</span> which means the policy is not a probability distribution of actions at $s$, but under $s$ we only execute a deterministic action, in another word, the probability execution of a chosen action $a=\mu_\theta(s)$ under $s$ is 1 and all other actions is 0.

Examine the gradient of the objective function that we have above:
$$
\begin{eqnarray}
\nabla_\theta J(\theta) =  E_{\tau\sim p_\theta(\tau), a\sim\pi_\theta}\\bigg[\nabla_\theta \log\pi_\theta(a|s)\\Big(Q^\pi(s,a)-V^\pi(s)\\Big)\\bigg]
\end{eqnarray}
$$
From the Bellman Equation we have the relationship between $Q^\pi$ and $V^\pi$, now the objective function becomes:
$$
\begin{eqnarray}
\nabla_\theta J(\theta) =  E_{\tau\sim p_\theta(\tau), a\sim \pi_\theta}\\bigg[\nabla_\theta \log\pi_\theta(a|s)\\Big(R + \gamma V^\pi(s_{t+1})- V^\pi(s)\\Big)\\bigg]
\end{eqnarray}
$$

The objective function depends on 2 things: policy $\pi_\theta$ and value function $V^\pi$. Assuming that we have an approximation function for $V^\pi(s)$ is $V_\phi(s)$ depending on parameters $\phi$.<br/>
We call the approximation function for policy $\pi_\theta$ is Actor and the approximation function for $V_\phi$ is Critic.

# 7 - Actor-Critic Algorithm
From REINFORCE algorithm, now we use an additional approximation function for value function $V_\phi$, changing a bit and we have:

Batch Actor-Critic:<br/>
    1. Sample a rollout $\tau$ to terminal state by the policy $\pi_\theta$<br/>
    2. Fit $V_\phi$ avec $y = \sum\_{i}^{T} r_i$<br/>
    3. Find $A(s_t,a_t) = r(s_t, a_t) + \gamma V_\phi(s_{t+1}) - V_\phi(s_{t})$<br/>
    4. Find $\nabla_\theta J(\theta) = \sum_i \nabla \log \pi_\theta (a_i|s_i) A^\pi (s_i, a_i)$<br/>
    5. Update $\theta \leftarrow \theta  + \alpha \nabla_\theta J(\theta)$<br/>
<br/>
<br/>
Above, we can represent $V_\phi(s) = r + V_\phi(s')$ according to Bellman Equation, therefore we could update the model knowing only 1 step ahead.<br/>
Online Actor-Critic:<br/>
    1. With policy $\pi_\theta$, execute 1 action $a \sim \pi_\theta(a|s)$ to have $(s,a,s',r)$<br/>
    2. Fit $V_\phi (s)$ avec $r + V_\phi(s')$<br/>
    3. Find $A(s_t,a_t) = r(s_t, a_t) + \gamma V_\phi(s_{t+1}) - V_\phi(s_{t})$<br/>
    4. Find $\nabla_\theta J(\theta) = \sum_i \nabla \log \pi_\theta (a_i|s_i) A (s_i, a_i)$<br/>
    5. Update $\theta \leftarrow \theta  + \alpha \nabla_\theta J(\theta)$<br/>
<br/>
<br/>
Thus, we update interactively both approximation functions $V_\phi$ and $\pi_\theta$.

# 8 - From Stochastic Actor-Critic to Q-Learning
Examine a policy as follow:
$$
\begin{eqnarray}
\pi'(a_t|s_t) = 1 \ \text{if}\  a_t = \arg \max_{a_t} A^\pi(s_t, a_t)
\end{eqnarray}
$$
Policy $\pi'$ is a Deterministic Policy: given a policy $\pi$ and assuming we know the Advantage of actions at state $s_t$ under policy $\pi$, we always choose the action with the highest Advantage at state $s$, probability of that action is 1, all other actions at $s_t$ is 0.
Policy $\pi'$ will be always better or at least equal to policy $\pi$. A policy is evaluated as equal or better than other if:
$V^\pi(s) \leq V^{\pi'} (s) \forall s \in S$ : with all state $s$ in the state space $S$, the return value $V^\pi(s)$ always less than or equal to the return value $V^{\pi'} (s)$.<br/>
For example, we have: at state $s$, we have 4 ways to go from state $s'$ corresponding to 4 actions and Advantages $A^\pi_1$, $A^\pi_2$, $A^\pi_3$, $A^\pi_4$. From state $s'$, we continue to follow policy $\pi$. From $s$ to $s'$, if we choose to follow stochastic policy $\pi$, expected Advantage is $\sum_{a \in A} p(a)A^\pi_a$, this quantity must be less than or equal to $\max_a A^\pi_a$.
<div style="width:image width px; font-size:80%; text-align:center;">
<img src="https://i.imgur.com/yMtTahR.jpg" align="center"/>
<div>Image 4: Transition from state $s$ to $s'$.</div>
</div>
</br>
Therefore, with a policy $\pi$, we always can apply policy $\pi'$ over it to have a new policy equal or better.<br/>
We now have the algorithm as follow:<br/>
1. Evaluate $A^\pi(s,a)$ with different actions $a$ <br/>
2. Optimize $\pi \leftarrow \pi'$

But evaluating $A^\pi(s,a)$ is also equivalent to evaluate $Q^\pi(s,a)$ because $A^\pi(s,a) =  Q^\pi(s,a) - V^\pi(s) = r(s,a)  + \gamma V^\pi(s') - V^\pi(s)$, and the quantity $V^\pi(s)$ is the same for different actions $a$ at state $s$.<br/>
The algorithm now becomes:
1. Evaluate $Q^\pi(s,a) \leftarrow r(s,a)  + \gamma V^\pi(s') $ with different actions $a$
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

# 9 - From Deep Q-Network to Deep Deterministic Policy Gradient

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
3. Evaluate $y_i \leftarrow r(s,a_i)  + \gamma Q_{\phi'}\\big(s', \mu_{\theta'}(s')\\big)$ <span class="tex2jax_ignore">(</span>use both policy and Q target network here<span class="tex2jax_ignore">)</span>
4. $\phi \leftarrow \phi - \alpha \frac{1}{N}\sum_i^N \frac{dQ_\phi}{d\phi}(s,a_i) \big(Q_\phi(s, a_i) - y_i\big)$
4. $\theta \leftarrow \theta - \beta \frac{1}{N}\sum_i^N \frac{d\mu_\theta}{d\theta}(s) \frac{dQ_\phi}{da}(s, a_i)$
5. Update target network $\phi' \leftarrow (1-\tau)\phi' + \tau \phi$ and $\theta' \leftarrow (1-\tau)\theta' + \tau \theta$
<br/>
<br/>
Notice in the  DDPG implementation:
- Because actions in DDPG are always deterministic, thus to explore the environment <span class="tex2jax_ignore">(</span>we do not want the agent always exploit the best trajectory in its knowledge, there may be a better path out there<span class="tex2jax_ignore">)</span>, a small action noise will be added into the action from the agent.
The noise in the [original paper](https://arxiv.org/abs/1509.02971) is a stochastic process named Ornstein–Uhlenbeck process <span class="tex2jax_ignore">(</span>OU process<span class="tex2jax_ignore">)</span>. The authors chose this process because the experiments gave good result, however other experiments conducted by different groups showing that other noise such as Gaussian Noise give the same performance.
- Implementation of action noise in the library [keras-rl](https://github.com/keras-rl/keras-rl) is OU process, however when I run, this noise is not decayed w.r.t time. Hoever, we need a big noise at the beginning <span class="tex2jax_ignore">(</span>to explore the environment<span class="tex2jax_ignore">)</span> and decrease after many episodes. This can be done if before we add the noise to the action, we multiply it with a quantity epsilon and the epsilon decays to $0$ w.r.t time.
- Besides adding noise into the action before executing it on the environment, they also add Gaussian Noise on the node of Neural Network. Reference paper [here](https://openai.com/blog/better-exploration-with-parameter-noise/). You could find the implementation o actor network noise in this library [stable baselines](https://stable-baselines.readthedocs.io/en/master/modules/ddpg.html#action-and-parameters-noise).

# 10 - Conclusion

To sum up, we went throught the Policy Gradient algorithm to DQN and DDPG.

Take away:
- Policy Gradient is an on-policy and a stochastic policy.
- Q-learning, DQN, DDPG are off-policy and deterministic policies.
- Always have a deterministic policy better than a stochastic policy.


