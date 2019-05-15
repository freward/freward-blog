<vue-mathjax></vue-mathjax>
# I - Reinforcement Learning - từ Policy Gradient đến Deep Deterministic Policy Gradient
Reinforcement Learning hay học củng cố/tăng cường, là lĩnh vực liên quan
đến việc dạy cho máy <span class="tex2jax_ignore">(</span>agent<span class="tex2jax_ignore">)</span> thực hiện tốt một nhiệm vụ <span class="tex2jax_ignore">(</span>task<span class="tex2jax_ignore">)</span> bằng cách
tương tác với môi trường <span class="tex2jax_ignore">(</span>environment<span class="tex2jax_ignore">)</span> thông qua hành động <span class="tex2jax_ignore">(</span>action<span class="tex2jax_ignore">)</span> và
nhận được phần thưởng <span class="tex2jax_ignore">(</span>reward<span class="tex2jax_ignore">)</span>. Cách học này rất giống với cách con người
học từ môi trường bằng cách thử sai. Lấy ví dụ 1 đứa vào mùa đông đến gần
lửa thì thấy ấm, đứa trẻ sẽ có xu hướng đến gần lửa nhiều hơn
<span class="tex2jax_ignore">(</span>vì nhận được phần thưởng là ấm áp<span class="tex2jax_ignore">)</span>,
nhưng chạm vào lửa thì nóng, đứa trẻ sẽ có xu hướng tránh chạm vào lửa <span class="tex2jax_ignore">(</span>vì
bị bỏng tay<span class="tex2jax_ignore">)</span>.

Trong ví dụ trên, phần thưởng xuất hiện ngay, việc điều chỉnh hành động là
tương đối dễ. Tuy nhiên, trong các tình huống phức tạp hơn khi mà phần thưởng
ở xa trong tương lai, điều này trở nên phức tạp hơn. Làm sao để đạt được tổng
phần thưởng cao nhất trong suốt cả quá trình? Reinforcement Learning <span class="tex2jax_ignore">(</span>RL<span class="tex2jax_ignore">)</span> là
các thuật toán để giải bài toán tối ưu này.

Dưới đây là định nghĩa của các thuật ngữ hay xuất hiện trong RL:

* _Environment_ <span class="tex2jax_ignore">(</span>môi trường<span class="tex2jax_ignore">)</span>: là không gian mà máy tương tác.
* _Agent_ <span class="tex2jax_ignore">(</span>máy<span class="tex2jax_ignore">)</span>: máy quan sát môi trường và sinh ra hành động tương ứng.
* _Policy_ <span class="tex2jax_ignore">(</span>chiến thuật<span class="tex2jax_ignore">)</span>: máy sẽ theo chiến thuật như thế nào để đạt được mục đích.
* _Reward_ <span class="tex2jax_ignore">(</span>phần thưởng<span class="tex2jax_ignore">)</span>: phần thưởng tương ứng từ môi trường mà máy nhận được khi thực hiện một hành động.
* _State_ <span class="tex2jax_ignore">(</span>trạng thái<span class="tex2jax_ignore">)</span>: trạng thái của môi trường mà máy nhận được.
* _Episode_ <span class="tex2jax_ignore">(</span>tập<span class="tex2jax_ignore">)</span>: một chuỗi các trạng thái và hành động cho đến trạng thái kết thúc $s_1,a_1,s_2,a_2,...s_T, a_T$
* _Accumulative Reward_ <span class="tex2jax_ignore">(</span>phần thưởng tích lũy<span class="tex2jax_ignore">)</span>: tổng phần thưởng tích lũy từ 1 state đến state cuối cùng.


<div style="width:image width px; font-size:80%; text-align:center;">
<img src="https://i.imgur.com/nIUdsIm.jpg" align="center"/>
<div>Hình 1: Vòng lặp tương tác giữa agent và environment.</div>
</div>
</br>

Như vậy, tại state $s$, agent tương tác với environment với hành động $a$,
dẫn đến state mới $s_{t+1}$ và nhận được reward tương ứng $r_{t+1}$.
Vòng lặp như thế cho đến trạng thái cuối cùng $s_T$.

Trong phần dưới đây, mình sẽ dùng các thuật ngữ tiếng Anh để các bạn tiện
theo dõi thay vì dịch sang tiếng Việt.
# 1 - Ví dụ
Xem ví dụ dưới đây từ openAI Gym, environment có tên
[MountaincontinuousCar-v0](https://github.com/openai/gym/wiki/MountainCarContinuous-v0).

<div style="width:image width px; font-size:80%; text-align:center;">
    <img src="https://i.imgur.com/yGWmDei.jpg" alt="MountaincontinuousCar-v0" style="padding-bottom:0.5em;" />
    <div>Hình 2: Một hình ảnh từ MountaincontinuousCar-v0.</div>
</div>
</br>

* _Goal_ : mục đích của bài toán là xây dựng policy để điều khiển xe lên đến được chỗ cắm cờ.
* _Environment_: dốc và xe chạy trong đó
* _State_: trạng thái của xe có 2 dimension, tọa độ của xe theo trục $x$ và vận tốc của xe tại thời điểm đo.
* _Action_: Lực được truyền cho xe để điều khiển, lực này không đủ mạnh
để đẩy xe 1 lúc lên đến cờ, xe sẽ cần đi qua đi lại 2 bên mặt nghiên để
tăng tốc đến chỗ cắm cờ.
* _Reward_: với mỗi step mà xe không đến được cờ, agent nhận reward $r=\frac{-a^2}{10}$,
xe đến được cờ thì nhận reward là 100. Như thế, nếu agent điều khiển xe mà xe không lên được thì sẽ bị phạt.
* _Terminal state_: nếu agent lên đến được cờ hoặc là số step vượt quá 998 steps.

# 2 - Policy Gradient
Trong phần dưới đây để ví dụ sinh động, mình giải quyết 1 bài toán game đơn giản, game Hứng Trứng.

<div style="width:image width px; font-size:80%; text-align:center;">
<img src="https://laptrinhcuocsong.com/images/game-hung-trung.png" align="center"/>
<div>Hình 3: Trò chơi Hứng Trứng.</div>
</div>
</br>

Gọi $\pi_\theta(a|s) = f(s, \theta)$ là policy của agent, đó hàm phân bố xác suất của action $a$ tại state $s$.

Trong game Hứng Trứng, giả sử ta có 3 action: qua trái, qua phải, đứng yên.
Tương ứng với state $s$ hiện tại <span class="tex2jax_ignore">(</span>vị trí của thùng hứng, vị trí của trứng rơi so với thùng,
tốc độ của bia rơi...<span class="tex2jax_ignore">)</span> ta sẽ có 1 phân bố xác suất của 3 action tương ứng,
ví dụ $[0.1, 0.3, 0.5]$. Tổng xác suất của tất cả các hành động tại state $s$ bằng $1$, ta có: $\sum_{a}\pi_\theta(a|s) = 1$.
Gọi $p(s_{t+1}|a_t, s_t)$ là hàm phân bố xác suất của state tiếp theo khi agent tại state $s$ và thực hiện action $a$.

Gọi $\tau = s_1, a_1, s_2, a_2,..., s_T, a_T$ là chuỗi sự kiện từ state $s_1$ đến state $s_T$. Xác suất xảy ra chuỗi $\tau$:

\[
\begin{eqnarray}
p_\theta(\tau) &=& p_\theta(s_1, a_1, s_2, a_2,...s_T, a_T) \\\\
               &=& p(s_1)\pi_\theta(a_1|s_1)p(s_2|s_1, a_1)\pi_\theta(a_2|s_2)...p(s_{T}|s_{T-1},a_{T-1})\pi_\theta(a_T|s_T) \\\\
               &=& p(s_1)\Pi_{t=1}^{t=T}\pi_\theta(a_t|s_t)p(s_{t+1}|s_t, a_t) \\\\
\end{eqnarray}
\]



Chúng ta sẽ thấy phân bố xác suất của state $p(s_{t+1}|a_t, s_t)$ bị loại bỏ trong các phương trình về sau.

**Mục tiêu của reinforcement learning là tìm $\theta$ sao cho:**

$$
\begin{eqnarray}
\theta^* &=& \arg\max_\theta E_{\tau\sim p_\theta(\tau)}\\big[r(\tau)\\big] \\\\
         &=& \arg\max_\theta E_{\tau\sim p_\theta(\tau)}\\bigg[\sum_t r(a_t, s_t)\\bigg]
\end{eqnarray}
$$

Từ công thức ta có thể thấy $\theta^\*$ là bộ tham số sao cho expectation của accumulative reward từ rất
nhiều các mẫu $\tau$ khác nhau có được từ việc thực thi theo policy $\pi_\theta$ là lớn nhất.<br />
Sau khi trải qua $N$ episodes khác nhau ta thu được $N$ mẫu $\tau$ khác nhau. Hàm số mục tiêu của bài toán lúc này:

$$
\begin{eqnarray}
J(\theta) &=& E_{\tau\sim p_\theta(\tau)}\\bigg[\sum_t r(a_t, s_t)\\bigg] \\\\
          &=& \frac{1}{N} \sum_i\sum_t r(a_t, s_t)
\end{eqnarray}
$$

$J(\theta)$ chính là trung bình cộng của accumulative reward của episodes khác nhau.<br/>
Chúng ta cũng có thể biểu diễn $J(\theta)$ theo phân bố của xác suất phân bố $p_\theta(\tau)$ như sau:

$$
\begin{eqnarray}
J(\theta) &=& E_{\tau\sim p_\theta(\tau)}\\bigg[\sum_t r(a_t, s_t)\\bigg] \\\\
          &=& \int p_\theta(\tau) r(\tau) dr
\end{eqnarray}
$$

Tiếp tục xem xét gradient của hàm mục tiêu:

$$
\begin{eqnarray}
\nabla_\theta J(\theta) &=& \nabla_\theta E_{\tau\sim p_\theta(\tau)}\\bigg[\sum_t r(a_t, s_t)\\bigg] \\\\
&=& \int \nabla_\theta p_\theta(\tau) r(\tau) dr
\end{eqnarray}
$$
Mà chúng ta lại có:
$$
\begin{eqnarray}
\nabla_\theta p_\theta(\tau) &=&  p_\theta(\tau) \frac{\nabla_\theta p_\theta(\tau)} {p_\theta(\tau)} \\\\
                             &=& p_\theta(\tau)\nabla_\theta \log p_\theta(\tau)
\end{eqnarray}
$$
**Lưu ý** trick này rất thường xuyên được sử dụng, do đó:
$$
\begin{eqnarray}
\nabla_\theta J(\theta) &=& \int p_\theta(\tau) \nabla_\theta \log p_\theta(\tau) r(\tau) dr \\\\
                        &=& E_{\tau\sim p_\theta(\tau)}\\bigg[\nabla_\theta \log p_\theta(\tau) r(\tau)\\bigg]
\end{eqnarray}
$$

Tiếp tục phân tích hàm $\log p_\theta(\tau)$, như ta đã có ở trên $p_\theta(\tau) = p(s_1)\Pi_{t=1}^{t=T}\pi_\theta(a_t|s_t)p(s_{t+1}|s_t, a_t)$, ta có:
$$
\begin{eqnarray}
\log p_\theta(\tau) = \log p(s_1) + \sum_{t=1}^{t=T}\log \pi_\theta(a_t|s_t) + \sum_{t=1}^{t=T}\log p(s_{t+1}|s_t, a_t)
\end{eqnarray}
$$

Cuối cùng:
$$
\begin{eqnarray}
\nabla_\theta \log p_\theta(\tau) = \sum_{t=1}^{t=T}\nabla_\theta \log \pi_\theta(a_t|s_t)
\end{eqnarray}
$$
Kết quả này rất hay vì đạo hàm theo $\theta$ của hàm $\log p_\theta(\tau)$ đã không còn phụ thuộc vào phân bố xác suất chuyển của state $p(s_{t+1}|a_t, s_t)$, nó chỉ phụ thuộc vào phân bố xác suất của action $a_i$ chúng ta thực hiện trên $s_i$.

Gradient của hàm mục tiêu lúc này:
$$
\begin{eqnarray}
\nabla_\theta J(\theta) &=&  E_{\tau\sim p_\theta(\tau)}\\bigg[\nabla_\theta \log p_\theta(\tau) r(\tau)\\bigg] \\\\
&=& E_{\tau\sim p_\theta(\tau)}\\bigg[\sum_{t=1}^{t=T}\nabla_\theta \log\pi_\theta(a_t|s_t)\sum_{t=1}^{t=T} r(a_t, s_t)\\bigg]
\end{eqnarray}
$$

Tương tự, sau khi trải qua $N$ episodes, expectation của hàm gradient này là:
$$
\begin{eqnarray}
\nabla_\theta J(\theta) &=& \frac{1}{N}\sum_{i=1}^{N}\\bigg(\sum_{t=1}^{t=T}\nabla_\theta \log \pi_\theta(a_{i,t}|s_{i,t})\\bigg)\\bigg(\sum_{t=1}^{t=T} r(a_{i,t}, s_{i,t})\\bigg)
\end{eqnarray}
$$

Cuối cùng, chúng ta update $\theta$ như dùng gradient ascent:
$$
\begin{eqnarray}
\theta \leftarrow \theta + \nabla_\theta J(\theta)
\end{eqnarray}
$$

# 3 - REINFORCE algorithm
Tổng hợp các kết quả trên ta có thuật toán REINFORCE như dưới đây:

1. Lấy 1 tập N chuỗi {$\tau^i$} dựa theo policy $\pi_\theta$
2. Tính gradient: $\nabla_\theta J(\theta) = \frac{1}{N}\sum_{i=1}^{N}\\bigg(\sum_{t=1}^{t=T}\nabla_\theta \log \pi_\theta(a_{i,t}|s_{i,t})\\bigg)\\bigg(\sum_{t=1}^{t=T} r(a_{i,t}, s_{i,t})\\bigg) $
3. Update $\theta \leftarrow \theta + \nabla_\theta J(\theta)$

Bây giờ hãy dừng lại để xem xét gradient của phương trình mục tiêu. Viết ở dạng đỡ rối mắt hơn ta có:

$$
\begin{eqnarray}
\nabla_\theta J(\theta) = \frac{1}{N}\sum_{i=1}^{N}\nabla_\theta \log \pi_\theta(\tau_i)r(\tau_i)
\end{eqnarray}
$$
Đây chính là maximum likelihood estimation [MLE](https://vi.wikipedia.org/wiki/H%E1%BB%A3p_l%C3%BD_c%E1%BB%B1c_%C4%91%E1%BA%A1i) tích với accumulative reward.
Việc tối ưu hàm mục tiêu cũng đồng nghĩa với việc tăng xác suất để đi theo chuỗi $\tau$ cho accumulative reward cao.

# 4 - Định nghĩa thêm 1 số khái niệm mới
$V^\pi(s)$: accumulative reward mong đợi tại state $s$ nếu đi theo policy $\pi$.<br/>
$Q^\pi(s,a)$: accumulative reward mong đợi nếu thực hiện action $a$ tại state $s$ nếu đi theo policy $\pi$.<br/>
Quan hệ giữa $V^\pi(s)$ và $Q^\pi(s,a)$: $V^\pi(s) = \sum_{a \in A}\pi_\theta(s,a)Q^\pi(s,a)$ - điều này là hợp lý vì $\pi_\theta(s,a)$ là xác suất thực hiện action $a$ tại $s$.<br/>
Ta cũng có như sau:
$$
\begin{eqnarray}
V^\pi(s_t) &=& E_\pi[G_t | S=s_t] \\\\
Q^\pi(s_t,a_t) &=& E_\pi[G_t|S=s_t, A=a_t]
\end{eqnarray}
$$
Trong đó:<br/>
$G_t=\sum^{\infty}\_{k=0}\gamma^kR_{k+t+1}$: tổng tất cả các reward nhận được kể từ state $s_t$ đến tương lai, với đại lượng $\gamma$ gọi là discount factor: $0 < \gamma < 1$. Càng xa hiện tại, reward càng bị discount nhiều, agent quan tâm nhiều hơn đến reward ở gần hơn là ở xa.

## 4.1 - Bellman Equations

Từ công thức ở trên, ta có:

$$
\begin{eqnarray}
V^\pi(s_t) &=& E_\pi\\bigg[G_t|S=s_t\\bigg] \\\\
           &=& E_\pi\\bigg[\sum^{\infty}\_{k=0}\gamma^kR_{k+t+1}|S=s_t\\bigg] \\\\
\end{eqnarray}
$$

Lấy reward $R_{t+1}$ nhận được khi chuyển từ state $s_t$ sang $s_{t+1}$ ra ngoài dấu $\sum$, ta được:

$$
\begin{eqnarray}
E_\pi\\bigg[R_{t+1} + \gamma\sum^{\infty}\_{k=0}\gamma^kR_{k+t+2}|S=s_t\\bigg] &=& E_\pi[R_{t+1}|S=s_t] + \gamma E_\pi\\bigg[\sum^{\infty}\_{k=0}\gamma^kR_{k+t+2}|S=s_t\\bigg]
\end{eqnarray}
$$

Khai triển expected value của 2 cụm ở trên ta có:

$$
\begin{eqnarray}
E_\pi\\bigg[R_{t+1}|S=s_t\\bigg]=\sum_a \pi(s_t,a) \sum_{s_{t+1}} p(s_{t+1}|s_t, a)R(s_{t+1}|s_t, a)
\end{eqnarray}
$$
Mà:

$$
\begin{eqnarray}
\gamma E_\pi\\bigg[\sum^{\infty}\_{k=0}\gamma^kR_{k+t+2}|S=s_t\\bigg] = \sum_a \pi(s_t,a) \sum_{s_{t+1}} p(s_{t+1}|s_t, a)\gamma E_\pi\\bigg[\sum^\infty_{k=0} \gamma^k R_{t+k+2} | S = s_{t+1}\\bigg]
\end{eqnarray}
$$
Ta có:
$$
\begin{eqnarray}
V^\pi(s_t) = \sum_a \pi(s_t,a) \sum_{s_{t+1}} p(s_{t+1}|s_t, a)\\Bigg[R(s_{t+1}|s_t, a) + \gamma E_\pi\bigg[\sum^\infty_{k=0} \gamma^k R_{t+k+2} | S = s_{t+1} \\bigg]\\Bigg]
\end{eqnarray}
$$
Để ý rằng:
$$
\begin{eqnarray}
E_\pi\\bigg[\sum^\infty_{k=0} \gamma^k R_{t+k+2} | S = s_{t+1}\\bigg] = V^\pi(s_{t+1})
\end{eqnarray}
$$

Cuối cùng ta có:
$$
\begin{eqnarray}
V^\pi(s_t) = \sum_a \pi(s_t,a) \sum_{s_{t+1}} p(s_{t+1}|s_t, a)\bigg[R(s_{t+1}|s_t, a) + \gamma  V^\pi(s_{t+1})\bigg]
\end{eqnarray}
$$
Tương tự với:
$$
\begin{eqnarray}
Q^\pi(s_t, a_t) = \sum_{s_{t+1}} p(s_{t+1}|s_t, a)\bigg[R(s_{t+1}|s_t, a) + \gamma \sum_{a_{t+1}} \pi(s_{t+1}, a_{t+1}) Q^\pi (s_{t+1}, a_{t+1}) \bigg]
\end{eqnarray}
$$
Mà theo quan hệ giữa $V^\pi$ và $Q^\pi$ ở trên, thì ta lại có:
$$
\begin{eqnarray}
\sum_{a_{t+1}} \pi(s_{t+1}, a_{t+1}) Q^\pi (s_{t+1}, a_{t+1}) = V^\pi(s_{t+1})
\end{eqnarray}
$$
Do đó:
$$
\begin{eqnarray}
Q^\pi(s_t, a_t) = \sum_{s_{t+1}} p(s_{t+1}|s_t, a_t)\\bigg[R(s_{t+1}|s_t, a_t) + \gamma  V^\pi(s_{t+1}) \\bigg]
\end{eqnarray}
$$

Tất cả những biến đổi ở trên cho thấy ta có thể biểu diễn giá trị của $Q^\pi$ và $V^\pi$ tại state $s_t$ với state $s_{t+1}$. Như vậy, nếu biết được giá trị tại state $s_{t+1}$, ta có thể dễ dàng tính toán được giá trị tại $s_t$. Tóm gọn thì ta có 2 phương trình sau:
$$
\begin{eqnarray}
V^\pi(s_t) &=& \sum_a \pi(s_t,a) \sum_{s_{t+1}} p(s_{t+1}|s_t, a)\\bigg[R(s_{t+1}|s_t, a) + \gamma  V^\pi(s_{t+1})\\bigg] \\\\
Q^\pi(s_t, a_t) &=& \sum_{s_{t+1}} p(s_{t+1}|s_t, a_t)\\bigg[R(s_{t+1}|s_t, a_t) + \gamma  V^\pi(s_{t+1}) \\bigg]
\end{eqnarray}
$$


Trở lại với gradient hàm mục tiêu, bây giờ ta có:
$$
\begin{eqnarray}
\nabla_\theta J(\theta) =  E_{\tau\sim p_\theta(\tau), a\sim\pi_\theta}\\bigg[\nabla_\theta \log\pi_\theta(a|s)Q^\pi(s,a)\\bigg]
\end{eqnarray}
$$

# 5 - Advantage <span class="tex2jax_ignore">(</span>lợi thế<span class="tex2jax_ignore">)</span>
$$
\begin{eqnarray}
\nabla_\theta J(\theta) =  E_{\tau\sim p_\theta(\tau), a\sim\pi_\theta}\\bigg[\nabla_\theta \log\pi_\theta(a|s)Q^\pi(s,a)\\bigg]
\end{eqnarray}
$$
Gradient của hàm mục tiêu cho thấy việc tăng khả năng thực hiện action $a$ nếu nhận được $Q^\pi(s,a)$ cao. Giả sử agent ở tại state $s$, việc ở tại state $s$ là đã có lợi cho agent rồi, thực hiện action $a$ nào cũng cho ra giá trị $Q^\pi(s,a)$ cao thì ta không thể phân tách <span class="tex2jax_ignore">(</span>discriminate<span class="tex2jax_ignore">)</span> các action $a$ với nhau và từ đó không biết được action $a$ nào là tối ưu. Do đó ta cần có 1 baseline để so sánh các giá trị $Q^\pi(s,a)$.<br/>
Như trong phần 4, ta có $V^\pi(s)$ là expectation của accumulative reward tại state $s$, không quan trọng tại $s$ agent thực hiện action gì, chúng ta mong đợi 1 accumulative reward là $V^\pi(s)$.
Do đó, 1 action $a_m$ được đánh giá là tệ nếu như $Q^\pi(s,a_m)$ < $V^\pi(s)$ và 1 action $a_n$ được đánh giá là tốt nếu như $Q^\pi(s,a_n)$ > $V^\pi(s)$. Từ đây ta có được 1 baseline để so sánh $Q^\pi(s,a)$ đó là $V^\pi(s)$. Gradient của objective function bây giờ có thể viết lại được như sau:
$$
\begin{eqnarray}
\nabla_\theta J(\theta) =  E_{\tau\sim p_\theta(\tau), a\sim\pi_\theta}\\bigg[\nabla_\theta \log\pi_\theta(a|s)\\Big(Q^\pi(s,a)-V^\pi(s)\\Big)\\bigg]
\end{eqnarray}
$$

Nếu $Q^\pi(s,a)-V^\pi(s) < 0$, 2 gradient ngược dấu với nhau, tối ưu hàm mục tiêu sẽ làm giảm gradient của việc thực thi hành động $a$ tại $s$.<br/>
Ta gọi $A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)$ là Advantage của action $a$ tại state $s$.

# 6 - Stochastic Actor-Critic
Stochastic Actor <span class="tex2jax_ignore">(</span>ngẫu nhiên Actor<span class="tex2jax_ignore">)</span> ý chỉ policy $\pi_\theta(a|s)$ là một hàm phân phối xác suất của action $a$ tại $s$. Ta gọi Stochastic Actor để phân biệt với Deterministic Actor <span class="tex2jax_ignore">(</span>hay Deterministic Policy<span class="tex2jax_ignore">)</span> mang ý chỉ policy không còn là một hàm phân phối xác suất của các action $a$ tại $s$, mà dưới $s$ ta chỉ thực hiện chính xác một action nhất định mà thôi $a=\mu_\theta(s)$, hay nói cách khác xác suất thực hiện $a$ tại $s$ bây giờ là 1.

Xem xét gradient của hàm mục tiêu mà ta đã có ở phần trên:
$$
\begin{eqnarray}
\nabla_\theta J(\theta) =  E_{\tau\sim p_\theta(\tau), a\sim\pi_\theta}\\bigg[\nabla_\theta \log\pi_\theta(a|s)\\Big(Q^\pi(s,a)-V^\pi(s)\\Big)\\bigg]
\end{eqnarray}
$$
Từ Bellman Equation ở trên ta có quan hệ giữa $Q^\pi$ và $V^\pi$, lúc này hàm mục tiêu trở thành:
$$
\begin{eqnarray}
\nabla_\theta J(\theta) =  E_{\tau\sim p_\theta(\tau), a\sim \pi_\theta}\\bigg[\nabla_\theta \log\pi_\theta(a|s)\\Big(R + \gamma V^\pi(s_{t+1})- V^\pi(s)\\Big)\\bigg]
\end{eqnarray}
$$

Hàm mục tiêu phụ thuộc vào 2 thứ: policy $\pi_\theta$ và value function $V^\pi$. Giả sử ta có một hàm xấp xỉ cho $V^\pi(s)$ là $V_\phi(s)$ phụ thuộc vào bộ tham số $\phi$.<br/>
Ta gọi hàm xấp xỉ cho policy $\pi_\theta$ là Actor và hàm xấp xỉ cho value function $V_\phi$ là Critic.

# 7 - Actor-Critic Algorithm
Từ thuật toán REINFORCE, bây giờ chúng ta sử dụng thêm hàm xấp xỉ cho value function $V_\phi$, thay đổi một chút ta có như sau:

Batch Actor-Critic:<br/>
    1. Lấy 1 chuỗi $\tau$ đến terminal state dựa theo policy $\pi_\theta$<br/>
    2. Fit $V_\phi$ với $y = \sum\_{i}^{T} r_i$<br/>
    3. Tính $A(s_t,a_t) = r(s_t, a_t) + \gamma V_\phi(s_{t+1}) - V_\phi(s_{t})$<br/>
    4. Tính $\nabla_\theta J(\theta) = \sum_i \nabla \log \pi_\theta (a_i|s_i) A^\pi (s_i, a_i)$<br/>
    5. Update $\theta \leftarrow \theta  + \alpha \nabla_\theta J(\theta)$<br/>
<br/>
<br/>
Mà ta có ở trên, ta có thể biểu diễn $V_\phi(s) = r + V_\phi(s')$ theo Bellman Equation, do đó ta có thể update model mà chỉ cần 1 step về phía trước.<br/>
Online Actor-Critic:<br/>
    1. Dựa theo policy $\pi_\theta$, thực hiện 1 action $a \sim \pi_\theta(a|s)$ để có $(s,a,s',r)$<br/>
    2. Fit $V_\phi (s)$ với $r + V_\phi(s')$<br/>
    3. Tính $A(s_t,a_t) = r(s_t, a_t) + \gamma V_\phi(s_{t+1}) - V_\phi(s_{t})$<br/>
    4. Tính $\nabla_\theta J(\theta) = \sum_i \nabla \log \pi_\theta (a_i|s_i) A (s_i, a_i)$<br/>
    5. Update $\theta \leftarrow \theta  + \alpha \nabla_\theta J(\theta)$<br/>
<br/>
<br/>
Như vậy, chúng ta cùng lúc update cả 2 hàm xấp xỉ $V_\phi$ và $\pi_\theta$.

# 8 - Từ Stochastic Actor-Critic tới Q-Learning
Xét một policy như sau:
$$
\begin{eqnarray}
\pi'(a_t|s_t) = 1 \ \text{if}\  a_t = \arg \max_{a_t} A^\pi(s_t, a_t)
\end{eqnarray}
$$
Policy $\pi'$ này là một Deterministic Policy: cho trước một policy $\pi$ và giả sử ta biết được Advantage của các action tại state $s_t$ dưới policy $\pi$, ta luôn chọn action cho ra giá trị Advantage lớn nhất tại state $s_t$ đó, probability của action đó là 1, tất cả các action khác tại $s_t$ bằng 0.
Policy $\pi'$ sẽ luôn tốt hơn hoặc ít nhất là tương đương với policy $\pi$. Một policy được đánh giá là tương đương hay tốt hơn khi ta có:
$V^\pi(s) \leq V^{\pi'} (s) \forall s \in S$ : với mọi state $s$ trong miền state $S$, giá trị return  $V^\pi(s)$ luôn nhỏ hơn hoặc bằng giá trị return $V^{\pi'} (s)$.<br/>
Ví dụ ta có như sau: tại state $s$, ta có 4 cách đi sang state $s'$ ứng với 4 action và ứng với $A^\pi_1$, $A^\pi_2$, $A^\pi_3$, $A^\pi_4$. Kể từ state $s'$, ta lại đi theo policy $\pi$. Từ $s$ sang $s'$, nếu chọn action theo stochastic policy $\pi$, Expected Advantage là $\sum_{a \in A} p(a)A^\pi_a$, lượng này chắc chắn nhỏ hơn $\max_a A^\pi_a$.
<div style="width:image width px; font-size:80%; text-align:center;">
<img src="https://i.imgur.com/yMtTahR.jpg" align="center"/>
<div>Hình 4: Thay đổi từ trạng thái $s$ sang $s'$.</div>
</div>
</br>
Như vậy, với một policy $\pi$, ta luôn có thể áp dụng policy $\pi'$ trên đó để được một policy ít nhất là bằng hoặc tốt hơn.<br/>
Ta có thuật toán bây giờ có thể viết như sau:<br/>
1. Đánh giá $A^\pi(s,a)$ với các action $a$ khác nhau<br/>
2. Tối ưu $\pi \leftarrow \pi'$

Mà đánh giá $A^\pi(s,a)$ thì cũng tương đương đánh giá $Q^\pi(s,a)$ vì $A^\pi(s,a) =  Q^\pi(s,a) - V^\pi(s) = r(s,a)  + \gamma V^\pi(s') - V^\pi(s)$, mà lượng $V^\pi(s)$ thì lại không đổi giữa các action $a$ tại state $s$.<br/>
Như vậy, thuật toán trở thành:
1. Đánh giá $Q^\pi(s,a) \leftarrow r(s,a)  + \gamma V^\pi(s') $ với các action $a$ khác nhau
2. Tối ưu $\pi \leftarrow \pi'$ : chọn ra action cho $A$ cao nhất, hay cũng chính là $Q$

Đến đây thì thực sự ta không cần quan tâm đến policy nữa, mà bước 2 có thể viết lại thành:
1. Đánh giá $Q^\pi(s,a) \leftarrow r(s,a)  + \gamma V^\pi(s') $ với các action $a$ khác nhau
2. $V^\pi(s) \leftarrow \max_a Q^\pi(s,a)$

Nếu xử dụng một hàm xấp xỉ cho $V_\phi(s)$, ta có thuật toán như sau:
1. Đánh giá $V^\pi(s) \leftarrow \max_a \big(r(s,a)  + \gamma V^\pi(s')\big)$
2. $\phi \leftarrow \arg min_\phi \big(V^\pi(s) - V_\phi(s)\big)^2$

Thuật toán này không ổn, ở bước 1 ta cần có reward $r(s, a)$ ứng với mỗi action $a$ khác nhau, như vậy ta cần nhiều simulation tại 1 state $s$. Ta có thể làm tương tự với $Q(s,a)$ thay vì $V(s)$.
1. Đánh giá $y_i \leftarrow r(s,a_i)  + \gamma \max_{a'} Q_\phi(s', a') $
2. $\phi \leftarrow \arg min_\phi \\big(Q_\phi(s, a_i) - y_i\\big)^2$ $(\*)$

Đây chính là thuật toán Q-Learning. Để ý rằng, reward $r$ ở trên không phụ thuộc vào state transition và cũng không
 phụ thuộc vào policy $\pi$ dùng để sinh ra sample, nên ta chỉ cần có sample $(s, a, r, s')$ là có thể cải thiện được
 policy mà không cần biết nó được sinh ra từ policy nào. Do đó, thuật toán này được gọi là off-policy. Sau này, chúng
 ta sẽ có các thuật toán on-policy, các thuật toán on-policy thì phải dựa vào sample được sinh ra tại policy hiện tại
 để có thể cải thiện policy mới.

Ta có thuật toán Online Q-Learning như sau:
1. Thực hiện action $a$ để có $(s, a, s', r)$
2. Đánh giá $y_i \leftarrow r(s,a_i)  + \gamma max_{a'} Q_\phi(s', a') $
3. $\phi \leftarrow \phi - \alpha \frac{dQ_\phi}{d\phi}(s,a) \big(Q_\phi(s, a_i) - y_i\big)$

Các bạn để ý bước 3, đó có phải là gradient descent như ở trên chỗ mình đánh dấu $(\*)$ không? Không, thực ra chúng ta đã lờ đi phần thay đổi của $y_i$ theo $\phi$ <span class="tex2jax_ignore">(</span>$y_i$ cũng phụ thuộc vào $\phi$<span class="tex2jax_ignore">)</span>. Như vậy, mỗi khi update $\phi$ theo thuật toán này, thì giá trị của mục tiêu $y_i$ cũng bị thay đổi theo! Mục tiêu luôn thay đổi khi ta cố gắng tiến gần lại nó, điều này làm cho thuật toán trở nên không ổn định.<br/>
Để giải quyết điều này, ta cần một hàm xấp xỉ khác gọi là target network, khác với train network chúng ta vẫn chạy. Target network sẽ được giữ cố định để tính $y$ và update dần dần.<br/>
Một vấn đề khác là các sample sinh ra liên tục nên nó rất liên quan <span class="tex2jax_ignore">(</span>correlation<span class="tex2jax_ignore">)</span> với nhau. Thuật toán trên cũng giống như Supervised Learning - ta map Q-value với giá trị mục tiêu, ta muốn các sample là độc lập <span class="tex2jax_ignore">(</span>i.i.d<span class="tex2jax_ignore">)</span> với nhau. Để phá vỡ correlation giữa các sample, ta có thể dùng 1 experience buffer: 1 list chứa rất nhiều sample $(s,a,r,s')$ khác nhau, và ta chọn ngẫu nhiên 1 batch từ buffer để train thuật toán.

Tóm lại, để thuật toán ổn định và hội tụ, ta cần:
- Hàm target network riêng biệt, gọi đó là $Q_{\phi'}$.
- Experience buffer.


Thuật toán bây giờ được viết như sau:
1. Thực hiện action $a_i$ để có $(s_i, a_i, s_i', r_i)$ và bỏ nó vào buffer.
2. Lấy ngẫu nhiên 1 batch $N$ sample từ buffer $(s_i, a_i, s_i', r_i)$.
3. Đánh giá $y_i \leftarrow r(s,a_i)  + \gamma max_{a'} Q_{\phi'}(s', a')$ <span class="tex2jax_ignore">(</span>dùng target network ở đây<span class="tex2jax_ignore">)</span>
4. $\phi \leftarrow \phi - \alpha \frac{1}{N}\sum_i^N \frac{dQ_\phi}{d\phi}(s,a_i) \big(Q_\phi(s, a_i) - y_i\big)$
5. Update target network $\phi' \leftarrow (1-\tau)\phi' + \tau \phi$ <span class="tex2jax_ignore">(</span>sử dụng $\tau \%$ của train network mới để update target network<span class="tex2jax_ignore">)</span>
<br/>
<br/>
Thuật toán này chính là Deep Q-Network <span class="tex2jax_ignore">(</span>DQN<span class="tex2jax_ignore">)</span>.

# 9 - Từ Deep Q-Network đến Deep Deterministic Policy Gradient

Thuật toán DQN đã thành công trong việc sấp xỉ Q-value, nhưng có một hạn chế ở bước 3: chúng ta cần đánh giá $Q_{\phi'}$ với tất cả các action khác nhau để chọn ra $Q$ lớn nhất. Với discrete action space như các game, khi mà action chỉ là các nút bấm lên xuống qua lại, số lượng action là hữu hạn, điều này có thể thực hiện được. Tuy nhiên, với continuous action space, ví dụ action có thể trong khoảng từ 0 đến 1, ta cần có một cách tiếp cận khác.

Cách ta có thể nghĩ đến đó là tìm cách phân nhỏ action space, phân nhỏ continuous space thành các khoảng nhỏ, ví dụ như từ 0 đến 1,  ta có thể phân ra làm 5 đến 10 khoảng giá trị có thể rơi vào. Một cách khác đó là sample ngẫu nhiên các action khác nhau trong khoảng cho phép với cùng state $s$ và chọn ra giá trị $Q(s,a)$ lớn nhất.

Deep Deterministic Policy Gradient <span class="tex2jax_ignore">(</span>DDPG<span class="tex2jax_ignore">)</span> có một cách tiếp cận tinh tế như sau, để ý rằng:
$$
\begin{eqnarray}
max_{a} Q_{\phi}(s, a) = Q_\phi\big(s, \arg max_a Q_\phi(s,a)\big)
\end{eqnarray}
$$

Bây giờ, nếu như ta có thêm một hàm xấp xỉ $\mu_\theta(s) = \arg max_a Q_\phi(s,a)$, bây giờ ta tìm bộ số $\theta$ sao cho: $\theta \leftarrow  \arg max_\theta Q_\phi(s,\mu_\theta(s))$. Phép tối ưu này xem xét sự thay đổi của $Q_\phi$ theo biến $\theta$. Ta có thể tính được sự thay đổi này dựa trên chain rule như sau:
$\frac{dQ_\phi}{d\theta} = \frac{dQ_\phi}{d\mu} \frac{d\mu}{d\theta}$.

Để ý rằng, $\mu_\theta(s)$ là một Deterministic Policy, chính vì vậy phương pháp này gọi là Deep Deterministic Policy Gradient.

Thuật toán DDPG như sau:

1. Thực hiện action $a_i$ để có $(s_i, a_i, s_i', r_i)$ và bỏ nó vào buffer.
2. Lấy ngẫu nhiên 1 batch $N$ sample từ buffer $(s_i, a_i, s_i', r_i)$.
3. Đánh giá $y_i \leftarrow r(s,a_i)  + \gamma Q_{\phi'}\\big(s', \mu_{\theta'}(s')\\big)$ <span class="tex2jax_ignore">(</span>dùng cả policy và Q target network ở đây<span class="tex2jax_ignore">)</span>
4. $\phi \leftarrow \phi - \alpha \frac{1}{N}\sum_i^N \frac{dQ_\phi}{d\phi}(s,a_i) \big(Q_\phi(s, a_i) - y_i\big)$
4. $\theta \leftarrow \theta - \beta \frac{1}{N}\sum_i^N \frac{d\mu_\theta}{d\theta}(s) \frac{dQ_\phi}{da}(s, a_i)$
5. Update target network $\phi' \leftarrow (1-\tau)\phi' + \tau \phi$ và $\theta' \leftarrow (1-\tau)\theta' + \tau \theta$
<br/>
<br/>
Lưu ý trong implement DDPG:
- Vì action trong DDPG luôn là deterministic, do đó để có thể khám phá môi trường <span class="tex2jax_ignore">(</span>ta không muốn agent luôn khai khác chỉ 1 đường đi tốt nhất trong những đường đi mà nó biết, có thể có đường đi khác tốt hơn mà nó chưa biết<span class="tex2jax_ignore">)</span>, chúng ta sẽ thêm vào action một lượng noise nhỏ vào action.
Lượng noise này trong [bài báo gốc](https://arxiv.org/abs/1509.02971) là một stochastic process có tên Ornstein–Uhlenbeck process <span class="tex2jax_ignore">(</span>OU process<span class="tex2jax_ignore">)</span>.
Người viết chọn process này vì khi làm thí nghiệm cho kết quả tôt, tuy nhiên một vài thí nghiệm khác sử dụng những noise khác như là 1
Gaussian Noise thì cũng cho kết quả tương đương.
- Implementation của action noise trong các thư viện như [keras-rl](https://github.com/keras-rl/keras-rl) là OU process, tuy nhiên khi chạy thử thì noise này không decay theo thời gian. Mà chúng ta cần noise lớn lúc ban đầu <span class="tex2jax_ignore">(</span>để khám phá môi trường nhiều<span class="tex2jax_ignore">)</span> sau đó giảm dần khi đã trải qua nhiều episode.
Việc này có thể thực hiện nếu trước khi thêm noise vào action, ta nhân nó với một lượng epsilon, mà epsilon giảm về xấp xỉ 0 theo thời gian.
- Ngoài việc thêm noise vào action trước khi thực hiện nó trên environment, người ta còn có thể thêm Gaussian Noise vào các nút mạng trong actor network.
Bài báo reference [tại đây](https://openai.com/blog/better-exploration-with-parameter-noise/). Bạn đọc có thể tìm thấy implementation của actor network noise tại thư viện [stable baselines](https://stable-baselines.readthedocs.io/en/master/modules/ddpg.html#action-and-parameters-noise).

# 10 - Kết bài

Như vậy, chúng ta đã di qua phần cơ bản từ thuật toán Policy Gradient đến DQN và DDPG.

Tóm gọn:
- Policy Gradient là một thuật toán on-policy và là một stochastic policy.
- Q-learning, DQN, DDPG là các thuật toán off-policy và là deterministic policy.
- Luôn có một deterministic policy tốt hơn một stochastic policy hiện có.


