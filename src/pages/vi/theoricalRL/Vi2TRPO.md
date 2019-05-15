<vue-mathjax></vue-mathjax>
# II - Stochastic Policy - từ Policy Gradient đến TRPO/PPO
Quay trở lại trong phần trước, ta có thuật toán REINFORCE Algorithm như sau:
1. Lấy 1 tập N chuỗi {$\tau^i$} dựa theo policy $\pi_\theta$
2. Tính gradient: $\nabla_\theta J(\theta) \approx \frac{1}{N}\sum_{i=1}^{N}\\bigg(\sum_{t=1}^{t=T}\nabla_\theta \log \pi_\theta(a_{i,t}|s_{i,t})\\bigg)\underbrace{\\bigg(\sum_{t=1}^{t=T} r(a_{i,t}, s_{i,t})\\bigg)}\_\text\{(\*)}$
3. Update $\theta \leftarrow \theta + \nabla_\theta J(\theta)$
<br/>
<br/>

# 1 - Causality - Quan hệ nhân quả
Bạn đọc hãy để ý phần dấu $(\*)$. Xét tại 1 thời điểm bất kì $t$, ta đã nhân MLE của action $a_t$ tại $s_t$ với tổng reward từ thời điểm $t=1$ cho đến hết episode $t=T$. Mà thực tế là hành động $a_t$ chỉ nên được đánh giá dựa trên tổng reward mà agent nhận được kể từ thời điểm $t$ trở về sau vì action $a_t$ không liên quan gì đến những reward thu được trước thời điểm $t$.<br/>
Do đó, thay vì ta lấy tổng reward từ $t=1$ đến $t=T$ thì ta chỉ nên tính tổng reward từ $t'=t$ đến $t=T$. Ta gọi đó là causality - quan hệ nhân quả.


[Thêm hình ở đây]

Thêm causality thì sẽ giúp ích cho việc giảm variance cho thuật toán và chúng ta sẽ có 1 đánh giá chính xác hơn về hệ quả của các action. Xem xét ví dụ sau, giả sử có 2 trajectory <span class="tex2jax_ignore">(</span>cách đi<span class="tex2jax_ignore">)</span> như hình minh họa:
- Cách 1 đi từ $s_1$ đến $s_t$ và nhận được tổng reward là 20, sau đó tại $s_t$ ta thực hiện hành động $a_{t_1}$, và từ đó cho đến kết thúc episode, ta nhận được reward là 10.
- Cách 2 cũng đi từ $s_1$ đến $s_t$ và nhưng nhận được tổng reward là 10, sau đó tại $s_t$ ta thực hiện hành động $a_{t_2}$, và từ đó cho đến kết thúc episode, ta nhận được reward là 20.

Rõ ràng cả 2 cách đi, tổng lượng reward từ $t=1$ đến $t=T$ đều là 30. Nếu ta không kể đến causality mà xét tại thời điểm $t'=t$, thì việc thực hiện hành động $a_{t_1}$ hay $a_{t_2}$ đều cho ta 1 tổng reward là 30 và theo công thức REINFORCE algorithm ban đầu, action $a_{t_1}$ và $a_{t_2}$ được đánh giá là như nhau. Nhưng thực tế là hành động $a_{t_2}$ tốt hơn và nên được ưu tiên thực hiện hơn nếu chúng ta đến $s_t$.

Do đó, REINFORCE algorithm lúc này trở thành:
1. Lấy 1 tập N chuỗi {$\tau^i$} dựa theo policy $\pi_\theta$
2. Tính gradient: $\nabla_\theta J(\theta) \approx \frac{1}{N}\sum_{i=1}^{N}\\bigg(\sum_{t=1}^{t=T}\nabla_\theta \log \pi_\theta(a_{i,t}|s_{i,t})\\bigg)\\bigg(\sum_{\color{red}{t'=t}}^{\color{red}{t=T}} r(a_{i,\color{red}{t'}}, s_{i,\color{red}{t'}})\\bigg)$
3. Update $\theta \leftarrow \theta + \nabla_\theta J(\theta)$

Lượng total reward mới được gọi là reward-to-go, tổng reward kể từ vị trí đó trở đi.

Xét riêng gradient của objective function ta có:

$$
\begin{eqnarray}
\nabla_\theta J(\theta) &\approx& \frac{1}{N}\sum_{i=1}^{N}\\bigg(\sum_{t=1}^{t=T}\nabla_\theta \log \pi_\theta(a_{i,t}|s_{i,t})\\bigg)\\bigg(\sum_{t'=t}^{t=T} r(a_{i,t'}, s_{i,t'})\\bigg) \\\\
&\approx& \frac{1}{N}\sum_{i=1}^{N}\\bigg(\sum_{t=1}^{t=T}\nabla_\theta \log \pi_\theta(a_{i,t}|s_{i,t})\\bigg)\hat{Q}^\pi_{i,t} \\\\
&\approx& \frac{1}{N}\sum_{i=1}^{N}\\bigg(\sum_{t=1}^{t=T}\nabla_\theta \log \pi_\theta(a_{i,t}|s_{i,t})\\bigg)\hat{A}^\pi_{i,t}
\end{eqnarray}
$$

Tóm tắt: như vậy, để giảm variance của REINFORCE algorithm, ta đã sử dụng 1 baseline cho total reward trong phần trước, và bây giờ sử dụng thêm tính chất causality. Kết hợp 2 nhận xét này lại, chúng ta đã có một REINFORCE algorithm cải thiện hơn nhiều so với bản đầu tiên, đánh giá action một cách chính xác hơn.

# 2 - Tiếp tục cải thiện Policy Gradient

Giả sử bây giờ chúng ta có 2 bộ parameters $\theta$ và ${\theta'}$ khác nhau cho 2 policy khác nhau: $\pi_\theta$ là policy cũ và $\pi_{\theta'}$ là policy mới. Làm sao để biết policy mới tốt hơn? Cách tốt nhất là xem thử policy nào cho giá trị objective function cao hơn, hay chúng ta đánh giá hiệu $J({\theta'}) - J(\theta)$. Từ đây trở đi mục đích của ta là tìm cách biểu diễn hiệu này.

Quay lại với objective function cơ bản:

$$
\begin{eqnarray}
J(\theta) &=& E_{\tau\sim p_\theta(\tau)}\\bigg[\sum_{t=0}^{t=T} r(a_t, s_t)\\bigg] \\\\
          &=& E_{s_0\sim p(s_0)}\\bigg[V^{\pi_\theta}(s_0)\\bigg]
\end{eqnarray}
$$

Nếu xét ngay từ state ban đầu $s_0$ và $s_0$ phân bố theo một phân bố xác suất là $p(s_0)$ thì ta có được $J(\theta)$ như trên, đó là expectation của value function tại state $s_0$ với policy là $\pi_\theta$. Một nhận xét khá tinh tế đó là: phân bố của $s_0$, state đầu tiên không phụ thuộc vào policy nào cả mà đó là một phân bố từ environment $(!)$. Do đó, từ $s_0$, cho dù ta có đi theo trajectory được sinh ra từ policy nào đi nữa, thì expectation của objective function với policy $\pi_\theta$ cũng là như nhau, giả sử ta sample trajectory từ một policy khác $\pi_{\theta'}$. Thể hiện bằng công thức:

$$
\begin{eqnarray}
J(\theta) &=& E_{s_0\sim p(s_0)}\\bigg[V^{\pi_\theta}(s_0)\\bigg] \\\\
          &=& E_{\tau\sim p_{\theta'}(\tau)}\\bigg[V^{\pi_\theta}(s_0)\\bigg]
\end{eqnarray}
$$

Quay lại với hiệu $J({\theta'}) - J(\theta)$, ta có như sau:

$$
\begin{eqnarray}
J(\theta') - J(\theta) &=& J(\theta') -  E_{\tau\sim p_{\theta'}(\tau)}\\bigg[V^{\pi_\theta}(s_0)\\bigg]\\\\
          &=& E_{\tau\sim p_{\theta'}(\tau)}\\bigg[V^{\pi_{\theta'}}(s_0)\\bigg] - E_{\tau\sim p_{\theta'}(\tau)}\\bigg[V^{\pi_\theta}(s_0)\\bigg] \\\\
          &=& E_{\tau\sim p_{\theta'}(\tau)}\\bigg[V^{\pi_{\theta'}}(s_0) - V^{\pi_\theta}(s_0)\\bigg] &&\color{red}{(\*)}
\end{eqnarray}
$$

Mà ta lại có:
$$
\begin{eqnarray}
V^{\pi_{\theta'}}(s_0) &=& \sum^\infty_{t=0}\gamma^t r(s_t, a_t) \\\\
V^{\pi_{\theta}}(s_0) &=& \sum^\infty_{t=0}\gamma^t V^{\pi_\theta}(s_t)-\sum^\infty_{t=1}\gamma^t V^{\pi_\theta}(s_t)\\\\
&=& \sum^\infty_{t=0}\gamma^t \Big(\gamma V^{\pi_\theta}(s_{t+1})-V^{\pi_\theta}(s_t)\Big)
\end{eqnarray}
$$

Phương trình thứ 2 thực tế ta chỉ thêm vào bớt ra cùng 1 lượng, nên 2 lượng trừ nhau đi nó cũng chỉ là $\gamma^0 V^{\pi_\theta}(s_0) = V^{\pi_\theta}(s_0)$. Thay 2 phương trình trên vào $\color{red}{(\*)}$ và đem $\sum$ ra ngoài, ta có:

$$
\begin{eqnarray}
J(\theta') - J(\theta)
          &=& E_{\tau\sim p_{\theta'}(\tau)}\\bigg[\sum^\infty_{t=0}\gamma^t \Big( r(s_t, a_t) + \gamma V^{\pi_\theta}(s_{t+1}) - V^{\pi_\theta}(s_t)\Big)\\bigg] \\\\
          &=& E_{\tau\sim p_{\theta'}(\tau)}\\bigg[\sum^\infty_{t=0}\gamma^t \Big( Q^{\pi_\theta}(s_t) - V^{\pi_\theta}(s_t)\Big)\\bigg] \\\\
\underbrace{J(\theta') - J(\theta)}\_\text\{(1)}
          &=& E_{\underbrace{\tau\sim p_{\theta'}(\tau)}\_\text\{(2)}}\\bigg[\sum^\infty_{t=0}\gamma^t \underbrace{A^{\pi_\theta}(a_t, s_t)}\_\text\{(3)}\\bigg] \\\\
\end{eqnarray}
$$

Bây giờ chúng ta hãy dừng lại để xem xét hiệu này 1 chút. Xét 3 cụm $(1)$, $(2)$ và $(3)$ trong phương trình trên.

- $(1)$ là sự chênh lệch về performance của policy mới $\pi_{\theta'}$ và policy cũ $\pi_{\theta'}$.
- $(2)$ là trajectrory được sinh ra từ policy mới $\pi_{\theta'}$.
- $(3)$ là Advantage của policy cũ $\pi_{\theta}$.

Hiệu $J(\theta') - J(\theta)$ đó là expectation của Advantage của policy cũ $\pi_{\theta}$ nhưng lại dưới những trajectory mới được sinh ra từ policy mới $\pi_{\theta'}$. Như vậy, cứ mỗi khi ta có 1 bộ $\theta'$ mới, ta sample trajectories từ bộ $\theta'$ mới này, sau đó mang những trajectory mới này để tính expectation của Advantage mà ta nhận được với bộ $\theta$ cũ, và đó chính là hiệu $J(\theta') - J(\theta)$.
Ta phải đoán mò giá trị $\theta'$ mới rồi mới có thể quay lại và xem thử nó tốt hơn cái cũ như thế nào. Nếu nó tốt hơn thì ta chọn, nếu không ta lại tìm bộ $\theta'$ khác.
Suy nghĩ kĩ một tí thì điều này khá là tự nhiên.


**nghĩ thêm minh họa và ví dụ ở đây**
có 1 chiến thuật A đã biết trước performance nó ntn dựa vào 1 bộ đánh giá. Giờ có 1 chiến thuật mới là B, xem thử B nó sẽ chọn đường đi nào. Sau đó dựa vào bộ đánh giá của A để xem B có tốt hơn A ko. Bình thường đi theo chiến thuật A thì chỉ trông đợi như thế thôi, giờ thử hành động của chiến thuật B, tự dưng thấy tòi ra 1 lượng Advantage.
**nghĩ thêm minh họa và ví dụ ở đây**

<br/>
Có cách nào để vẫn sử dụng những trajectory từ policy cũ $\pi_\theta$ mà vẫn đo được lượng thay đổi về performance này không? Nếu như thế thì ta không cần phải lấy sample từ policy mới $\pi_{\theta'}$ nữa. Để có được điều đó thì ta tìm cách khử dấu $'$ trong cụm $(2)$.<br/>
Công cụ dưới đây sẽ giúp giải quyết điều đó.

## 2.1 - Công cụ toán - Importance Sampling

Xét expectation của hàm $f(x)$ khi mà $x\sim p(x)$. Ta có:

$$
\begin{eqnarray}
E_{x \sim p(x)}[f(x)] &=& \int p(x) f(x) dx
\end{eqnarray}
$$

Giả sử có 1 hàm phân bố khác của $x$ là $x \sim q(x)$.
Nhân và chia phương trình trên cho cùng 1 lượng $q(x)$, ta có như sau:

$$
\begin{eqnarray}
E_{x \sim p(x)}[f(x)] &=& \int p(x) f(x) dx \\\\
&=& \int q(x) \frac{p(x)}{q(x)}f(x)dx \\\\
E_{\color{red}{x \sim p(x)}}[f(x)] &=& E_{\color{red}{x \sim q(x)}}\\bigg[ \frac{p(x)}{q(x)}f(x)\\bigg]
\end{eqnarray}
$$


Quay trở lại với phương trình hiệu $J(\theta)-J(\theta')$ ở trên, ta muốn khử đi dấu $'$:

$$
\begin{eqnarray}
J(\theta') - J(\theta)
          &=& E_{\tau\sim p_{\theta'}(\tau)}\\bigg[\sum^\infty_{t=0}\gamma^t A^{\pi_\theta}(a_t, s_t)\\bigg] \\\\
&=& \sum_t E_{s_t \sim p_{\theta'}(s_t), a_t \sim \pi_{\theta'}(a_t | s_t)}\\bigg[\gamma^t A^{\pi_\theta}(a_t, s_t)\\bigg] \\\\
&=& \sum_t E_{\underbrace{s_t \sim p_{\theta'}(s_t)}\_\text\{(B)}} \underbrace{\\Bigg[E_{a_t \sim \pi_{\theta'}(a_t | s_t)} \\bigg[\gamma^t A^{\pi_\theta}(a_t, s_t)\\bigg]\\Bigg]}\_\text\{(A)}
\end{eqnarray}
$$

Cụm $(A)$ có thể bỏ được dấu $'$ nhờ important sampling, ta có:

$$
\begin{eqnarray}
J(\theta') - J(\theta)
&=& \sum_t E_{s_t \sim p_{\theta'}(s_t)}\\Bigg[E_{a_t \sim \pi_{\theta'}(a_t | s_t)} \\bigg[\gamma^t A^{\pi_\theta}(a_t, s_t)\\bigg]\\Bigg] \\\\
&=& \sum_t E_{s_t \sim p_{\theta'}(s_t)}\\Bigg[E_{a_t \sim \pi_{\theta}(a_t | s_t)} \\bigg[\frac{\pi_{\theta'}(a_t | s_t)}{\pi_{\theta}(a_t | s_t)} \gamma^t A^{\pi_\theta}(a_t, s_t)\\bigg]\\Bigg]
\end{eqnarray}
$$

Khi ta thay đổi policy, thì phân bố của state $s_t$ cũng thay đổi theo. Cụm $(B)$ cho thấy điều đó. Nếu ta có thể bỏ được dấu $'$ kia đi thì rất tuyệt vời.
Bằng trực giác, ta có thể cảm nhận rằng, nếu $p_{\theta'}({s_t})$ không quá khác $p_\theta({s_t})$ nếu thì policy $\pi_{\theta'}$ không quá khác $\pi_{\theta}$. Nhưng làm sao để biết thế nào là không quá khác, làm sao để biết thay đổi từ $\pi_{\theta}$ thành $\pi_{\theta'}$ là đủ nhỏ để có thể xem $p_\theta({s_t})$ là không đổi?

# 3 - Tìm kiếm một bound thích hợp để thay đổi Policy

# 3.1 - Trường hợp Deterministic Policy

Xét trường hợp các policy $\pi_\theta$ deterministic và $a_t = \pi_\theta(s_t)$. $\pi_{\theta'}$ được xem là gần với $\pi_\theta$ nếu ta có:

$$
\begin{eqnarray}
\pi_{\theta'}(a_t \neq \pi_\theta(s_t) | s_t) \leq \epsilon \\;\\;\\;\\; \forall s_t \in S
\end{eqnarray}
$$

Xác suất để action $a_t$ của policy mới $\pi_{\theta'}$ khác với action $a_t$ của policy cũ $\pi_{\theta}$ tại $s_t$ là 1 đại lượng nhỏ hơn $\epsilon$.
Với mỗi state kể từ đầu cho đến $s_t$, xác xuất để 2 action giống nhau tại mỗi state là $1 - \epsilon$ là 2 action khác nhau là $\epsilon$. Như vậy, xác xuất để ta thực hiện y hệt chuỗi action từ state đầu đến $s_t$ là $(1-\epsilon)^t$ và xác xuất để ta thực hiện ít nhất khác 1 action trong chuỗi từ state đầu đến $s_t$ là $1-(1-\epsilon)^t$. Phân bố mới của $s_t$ dưới $\pi_{\theta'}$ như sau:

$$
\begin{eqnarray}
p_{\theta'}(s_t) = (1-\epsilon)^t p_\theta(s_t) + \big(1-(1-\epsilon)^t\big) p_{\text{diff}}(s_t)
\end{eqnarray}
$$

Xem xét sự khác nhau giữa 2 phân bố $p_{\theta'}(s_t)$ và $p_{\theta}(s_t)$:

$$
\begin{eqnarray}
\lvert p_{\theta'}(s_t) - p_{\theta}(s_t) \rvert &=& \lvert (1-\epsilon)^t p_\theta(s_t) + \big(1-(1-\epsilon)^t\big) p_{\text{diff}}(s_t) - p_{\theta}(s_t)\rvert \\\\
&=& \lvert \big((1-\epsilon)^t-1\big) p_\theta(s_t) + \big(1-(1-\epsilon)^t\big) p_{\text{diff}}(s_t) \rvert \\\\
&=& \big(1-(1-\epsilon)^t\big)\lvert p_{\text{diff}}(s_t) - p_\theta(s_t)\rvert \\\\
&\leq& 2\big(1-(1-\epsilon)^t\big) \\\\
&\leq& 2\epsilon t
\end{eqnarray}
$$

Ở đây ta đã dùng 2 cái bất phương trình:

$$
\begin{eqnarray}
\lvert \big((1-\epsilon)^t-1\big) p_\theta(s_t) + \big(1-(1-\epsilon)^t\big) p_{\text{diff}}(s_t) \rvert &\leq& 2 \\\\
1-\epsilon t &\leq& (1-\epsilon)^t \\;\\;\\;\\; \epsilon \in [0,1]
\end{eqnarray}
$$

Tóm lại, nếu như 2 policy khác nhau 1 lượng ít hơn hoặc bằng $\epsilon$ thì phân bố của state $s_t$ khác nhau 1 lượng ít hơn hoặc bằng $\epsilon t$ với $t$ là timestep đang xét. Ta có thể thấy, mặc dù xác suất khác nhau của 2 policy luôn nhỏ hơn hoặc bằng $\epsilon$, nếu t càng tăng <span class="tex2jax_ignore">(</span>càng xa so với gốc<span class="tex2jax_ignore">)</span> thì bound về độ khác nhau của phân bố state $s_t$ càng tăng.

# 3.2 - Trường hợp Stochastic Policy

Policy $\pi_\theta$ được xem là gần với policy $\pi_{\theta'}$ nếu:

$$
\begin{eqnarray}
\lvert \pi_{\theta'}(a_t|s_t) - \pi_{\theta}(a_t|s_t) \rvert \leq \epsilon
\\;\\;\\;\\; s_t \in S
\end{eqnarray}
$$

Nhưng mà policy $\pi_\theta$ là 1 stochastic policy, từ 1 distribution ta sample ra 1 action ngẫu nhiên, và tương tự với policy $\pi_{\theta'}$. Trong Computer Science, một chuỗi số ngẫu nhiên không được sinh ra một cách hoàn toàn ngẫu nhiên. Người ta dùng 1 số được gọi là random seed và bắt đầu sample với số random seed đó. Với cùng 1 số random seed thì chuỗi số ngẫu nhiên được sinh ra là hoàn toàn như nhau.
Như vậy ở đây ta xem như 2 policy $\pi_\theta$ và $\pi_{\theta'}$ được sample với cùng random seed thì xác suất để 2 action sai khác nhau là $\epsilon$. Đây chính là intuition của lemma dưới đây:

Nếu $\lvert p_{X}(x) - p_{Y}(x) \rvert \leq \epsilon$ thì tồn tại $p(x,y)$ mà $p(x)=p_X(x)$ và $p(y)=p_Y(y)$ và $p(x=y) = 1-\epsilon$.
Nghĩa là $p_X(x)$ đồng thuận với $p_Y(y)$ với xác suất là $1 - \epsilon$.
Do đó, $\pi_{\theta'}(a_t|s_t)$ thực hiện action khác với $\pi_{\theta}(a_t|s_t)$ có xác suất nhiều nhất là $\epsilon$.

Cuối cùng, ta cũng có 1 bound cho sự khác nhau giữa  2 phân bố $p_{\theta'}(s_t)$ và $p_{\theta}(s_t)$ như trên:

$$
\begin{eqnarray}
\lvert p_{\theta'}(s_t) - p_{\theta}(s_t) \rvert &=& \big(1-(1-\epsilon)^t\big)\lvert p_{\text{diff}}(s_t) - p_\theta(s_t)\rvert \\\\
&\leq& 2\big(1-(1-\epsilon)^t\big) \\\\
&\leq& 2\epsilon t
\end{eqnarray}
$$


Như vây, ta đã có được 1 cái bound cho sự khác nhau giữa 2 phân bố của state với $\theta$ và $\theta'$. Bây giờ, ta sẽ xem xét expectation của một làm $f(x)$ bất kì dưới $s_t \sim p_{\theta'}(s_t)$ như thế nào nhé.

$$
\begin{eqnarray}
E_{s_t \sim p_{\color{red}{\theta'}}(s_t)}[f(s_t)] = \sum_{s_t} p_{\theta'}(s_t) f(s_t) &\geq& \sum_{s_t} p_{\theta}(s_t) f(s_t) - \lvert p_{\theta'}(s_t) - p_{\theta}(s_t) \rvert \max_{s_t} f(s_t) \\\\
&\geq& E_{s_t \sim p_{\color{red}{\theta}}(s_t)}[f(s_t)] - 2\epsilon t \max_{s_t} f(s_t)
\end{eqnarray}
$$

Bây giờ ta đã có tất cả mọi thứ cần thiết. Quay trở lại với hiệu $J(\theta') - J(\theta)$:

$$
\begin{eqnarray}
J(\theta') - J(\theta) &=& \sum_t E_{s_t \sim p_{\color{red}{\theta'}}(s_t)}\\Bigg[E_{a_t \sim \pi_{\theta}(a_t | s_t)} \\bigg[\frac{\pi_{\theta'}(a_t | s_t)}{\pi_{\theta}(a_t | s_t)} \gamma^t A^{\pi_\theta}(a_t, s_t)\\bigg]\\Bigg] \\\\
&\geq& \sum_t E_{s_t \sim p_{\color{red}{\theta}}(s_t)}\\Bigg[E_{a_t \sim \pi_{\theta}(a_t | s_t)} \\bigg[\frac{\pi_{\theta'}(a_t | s_t)}{\pi_{\theta}(a_t | s_t)} \gamma^t A^{\pi_\theta}(a_t, s_t)\\bigg]\\Bigg] - \sum_t 2\epsilon t \max_{s_t} f(s_t) \\\\
&\geq& \sum_t E_{s_t \sim p_{\color{red}{\theta}}(s_t)}\\Bigg[E_{a_t \sim \pi_{\theta}(a_t | s_t)} \\bigg[\frac{\pi_{\theta'}(a_t | s_t)}{\pi_{\theta}(a_t | s_t)} \gamma^t A^{\pi_\theta}(a_t, s_t)\\bigg]\\Bigg] - 2\epsilon t C
\end{eqnarray}
$$

Với $C$ có bound là $t r_\max$ hoặc $t \frac{r_\max}{1-\gamma}$.

Chúng ta đã đến đích rồi: <br/>
Ta chỉ cần tối ưu lượng $\sum_t E_{s_t \sim p_{\color{red}{\theta}}(s_t)}\\Bigg[E_{a_t \sim \pi_{\theta}(a_t | s_t)} \\bigg[\frac{\pi_{\theta'}(a_t | s_t)}{\pi_{\theta}(a_t | s_t)} \gamma^t A^{\pi_\theta}(a_t, s_t)\\bigg]\\Bigg]$ thì hiệu $J(\theta') - J(\theta)$ cũng sẽ được tối ưu vì đó là bound dưới của hiệu $J(\theta') - J(\theta)$.

# 4 - Kullback–Leibler divergence <span class="tex2jax_ignore">(</span>KL divergence<span class="tex2jax_ignore">)</span>

Mình khuyến khích bạn đọc tìm hiểu về KL divergence tại [thetalog](https://thetalog.com/statistics/ly-thuyet-thong-tin/). Bài viết của bạn Tiến rất tuyệt vời và dễ hiểu.

Để đo độ sai khác giữa 2 phân bố policy, ta có thể dùng KL divergence. Ta có:

$$
\begin{eqnarray}
\lvert \pi_{\theta'}(s_t) - \pi_{\theta}(s_t) \rvert &\leq& \sqrt{\frac{1}{2}D_{KL}(\pi_{\theta'}(a_t|s_t) || \pi_{\theta}(a_t|s_t))}
\end{eqnarray}
$$

# 4 - Trust Region Policy Optimization <span class="tex2jax_ignore">(</span>TRPO<span class="tex2jax_ignore">)</span>

Tổng hợp tất cả kết quả ở trên thì:

$$
\begin{eqnarray}
&& \theta' \leftarrow \arg \max_{\theta'} \sum_t E_{s_t \sim p_{\theta}(s_t)}\\Bigg[E_{a_t \sim \pi_{\theta}(a_t | s_t)} \\bigg[\frac{\pi_{\theta'}(a_t | s_t)}{\pi_{\theta}(a_t | s_t)} \gamma^t A^{\pi_\theta}(a_t, s_t)\\bigg]\\Bigg] \\\\
&& \text{Với điều kiện là } D_{KL}(\pi_{\theta'}(a_t|s_t) || \pi_{\theta}(a_t|s_t)) \leq \epsilon
\end{eqnarray}
$$


Với $\epsilon$ đủ nhỏ thì điều này sẽ đảm bảo cải thiện $J(\theta') - J(\theta)$.

Bạn đọc có thấy ở trên mình đang tối ưu một hàm với điều kiện không? Mình khuyến khích bạn đọc tìm hiểu thêm về đối ngẫu Lagrange tại bài viết trên *machine learning cơ bản* của anh Vũ Hữu Tiệp [tại đây](https://machinelearningcoban.com/2017/04/02/duality/).

Thuật toán Trust Region Policy Optimization:

$$
\begin{eqnarray}
\mathcal{L}(\theta', \lambda)  = \sum_t E_{s_t \sim p_{\theta}(s_t)}\\Bigg[E_{a_t \sim \pi_{\theta}(a_t | s_t)} \\bigg[\frac{\pi_{\theta'}(a_t | s_t)}{\pi_{\theta}(a_t | s_t)} \gamma^t A^{\pi_\theta}(a_t, s_t)\\bigg]\\Bigg] - \lambda \big(D_{KL}(\pi_{\theta'}(a_t|s_t) || \pi_{\theta}(a_t|s_t)) - \epsilon\big)
\end{eqnarray}
$$
1. Maximize $\mathcal{L}(\theta', \lambda)$ theo $\theta'$.
2. $\lambda \leftarrow \lambda + \alpha \big(D_{KL}(\pi_{\theta'}(a_t|s_t) || \pi_{\theta}(a_t|s_t))- \epsilon\big)$

$\lambda$ được gọi là nhân tử Lagrange.
Mình có thể hiểu như sau: nếu $D_{KL}$ lớn hơn $\epsilon$, constraint bị vi phạm nhiều, hiệu $D_{KL} - \epsilon$ sẽ dương to, hàm cần optimize sẽ bị phạt nhiều, đồng thời $\lambda$ cũng được nâng lên <span class="tex2jax_ignore">(</span>ngược lại thì ta hạ xuống<span class="tex2jax_ignore">)</span>.

Thuật toán này được gọi là Trust Region vì ta chỉ thay đổi $\theta$ trong 1 miền nhỏ mà ở đó ta giả sử như phân bố của state với $\theta$ và $\theta'$ là như nhau, miền đó là miền mà ta tin tưởng là thuật toán sẽ chỉ có thể tốt lên với $\epsilon$ đủ nhỏ.

# 5 - Phụ thêm:
