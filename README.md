# ピアノ弦のシミュレーション

ピアノ弦の波動方程式として以下を採用し，ハンマーの打撃を $f(x, t)$ に入力する．

$$ \frac{\partial^2 y}{\partial t^2} = c^2 \frac{\partial^2 y}{\partial x^2} - \varepsilon c^2 L^2 \frac{\partial^4 y}{\partial x^4} - 2 b_1 \frac{\partial y}{\partial t} + 2b_3 \frac{\partial^3 y}{\partial t^3} + f(x, t) $$

詳細は以下を参照：

[1] Antoine Chaigne, Anders Askenfelt; Numerical simulations of piano strings. I. A physical model for a struck string using finite difference methods. J. Acoust. Soc. Am. 1 February 1994; 95 (2): 1112–1118.

[2] Antoine Chaigne, Anders Askenfelt; Numerical simulations of piano strings. II. Comparisons with measurements and systematic exploration of some hammer‐string parameters. J. Acoust. Soc. Am. 1 March 1994; 95 (3): 1631–1640.


同時に，ピアノ弦の周りの空気もシミュレートする．空気の圧力の変位は以下の減衰項付きの波動方程式によって計算し， $f(x,t)$ は $\displaystyle \frac{\partial^2 y}{\partial t^2}$ に比例して入力する．

$$ \frac{\partial^2 p}{\partial t^2} = c^2 \nabla^2 p - \gamma\frac{\partial p}{\partial t}+ f(x, t) $$

部屋の次元は選択可能だが，2次元で十分だと思われる．