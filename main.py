import json

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
from scipy.io.wavfile import write
from scipy.signal import resample
from tqdm import tqdm


class PianoString:
    """
    単一の弦(C4など)を有限差分でシミュレートするクラス
    """

    def __init__(self, config):
        with open(config, "r") as f:
            params = json.load(f)
        self.__dict__.update(params)

        # -- 弦の派生パラメータ
        self.mu = self.Ms / self.L  # 線密度
        self.c = np.sqrt(self.T / self.mu)  # 波の位相速度
        self.dt = 1 / self.fe  # サンプリング周期

        self.dx = self.L / self.N

        # ハンマーの位置
        self.i0 = int(self.N * self.alpha)

        # ハンマーの窓関数（例: ガウス窓）
        g = np.array(
            [
                np.exp(-(((i - self.i0) * self.dx / (self.span / 2)) ** 2))
                for i in range(self.N + 1)
            ]
        )
        # 数値計算の都合上，両端に0を追加
        g = np.concatenate([[0], g, [0]])
        g /= np.sum(g * self.dx)
        self.g = g

        # シミュレーションの初期化
        # 弦の各位置の高さ(3時刻)
        y = np.zeros((3, self.N + 3))
        # 打点でのハンマーの位置(2時刻)
        eta = np.zeros(2)
        # ハンマーの力
        Fh = 0
        # ハンマーが接触中かどうか
        hammer_contact = True
        # 時刻
        self.t = 0
        # 速度の記録
        self.mic_idx = int(self.N / 2)
        self.u_rec = [0] * self.u_rec_num

        dt = self.dt
        # step 1
        y[1, 1:-1] = (y[0, 2:] + y[0, :-2]) / 2
        y[1, [0, -1]] = 0

        eta[1] = self.Vh * dt
        Fh = self.K * np.power(np.abs(eta[0] - y[1, self.i0]), self.p)

        # step 2
        """
        y[2, 1:-1] = (
            y[1, 2:]
            + y[1, :-2]
            - y[0, 1:-1]
            + (dt**2 * self.N * Fh * self.g[1:-1] * self.N) / self.Ms
        )"""
        acc = Fh * self.g / self.Ms
        Dinv = 1 / (1 / dt**2 + 2 * self.b1 / dt)
        y[2, 1:-1] = Dinv * (
            self.c**2 * (y[1, 2:] - 2 * y[1, 1:-1] + y[1, :-2]) / self.dx**2
            - 2 * self.b1 * (-y[1, 1:-1]) / dt
            - (-2 * y[1, 1:-1] + y[0, 1:-1]) / dt**2
            + acc[1:-1]
        )
        y[2, [0, -1]] = 0
        eta_next = 2 * eta[1] - eta[0] - (dt**2 * Fh) / self.Mh
        Fh = self.K * np.power(np.abs(eta_next - y[2, self.i0]), self.p)
        eta[0], eta[1] = eta[1], eta_next
        self.Fh = Fh

        self.y = y
        self.eta = eta
        self.hammer_contact = hammer_contact

        self.u_rec[-2] = (y[1, self.mic_idx] - y[0, self.mic_idx]) / dt
        self.u_rec[-1] = (y[2, self.mic_idx] - y[1, self.mic_idx]) / dt

        # 方程式の係数
        """
        D = 1 + self.b1 * dt + 2 * self.b3 / dt
        r = self.c * dt / self.dx
        self.a1 = (2 - 2 * r**2 + self.b3 / dt - 6 * self.eps * self.N**2 * r**2) / D
        self.a2 = (-1 + self.b1 * dt + 2 * self.b3 / dt) / D
        self.a3 = r**2 * (1 + 4 * self.eps * self.N**2) / D
        self.a4 = (self.b3 / dt - self.eps * self.N**2 * r**2) / D
        self.a5 = (-self.b3 / dt) / D
        """

    def step(self, source=None):
        y = self.y
        eta = self.eta
        dt = self.dt

        if source is None:
            S = np.zeros(self.N + 3)
        else:
            S = source

        self.hammer_contact &= eta[1] > y[2, self.i0]

        if self.hammer_contact:
            acc = self.Fh * self.g / self.Ms
        else:
            acc = np.zeros(self.N + 3)

        y_next = np.zeros(self.N + 3)
        """
        y_next[2:-2] = (
            self.a1 * y[2, 2:-2]
            + self.a2 * y[1, 2:-2]
            + self.a3 * (y[2, 3:-1] + y[2, 1:-3])
            + self.a4 * (y[2, 4:] + y[2, :-4])
            + self.a5 * (y[1, 3:-1] + y[1, 1:-3] + y[0, 2:-2])
            + acc[2:-2]
        )
        """
        Dinv = 1 / (1 / dt**2 + 2 * self.b1 / dt - 2 * self.b3 / dt**3)
        y_next[2:-2] = Dinv * (
            self.c**2 * (y[2, 3:-1] - 2 * y[2, 2:-2] + y[2, 1:-3]) / self.dx**2
            - (self.eps * self.c**2 * self.L**2)
            * (y[2, 4:] - 4 * y[2, 3:-1] + 6 * y[2, 2:-2] - 4 * y[2, 1:-3] + y[2, :-4])
            / self.dx**4
            - 2 * self.b1 * (-y[2, 2:-2]) / dt
            + 2 * self.b3 * (-3 * y[2, 2:-2] + 3 * y[1, 2:-2] - y[0, 2:-2]) / dt**3
            - (-2 * y[2, 2:-2] + y[1, 2:-2]) / dt**2
            + acc[2:-2]
            + S[2:-2]
        )
        # 境界条件
        y_next[0], y_next[1] = -y_next[1], 0
        y_next[-2], y_next[-1] = 0, -y_next[-3]

        # ハンマー側の更新
        if self.hammer_contact:
            eta_next = 2 * eta[1] - eta[0] - (dt**2 * self.Fh) / self.Mh
            eta[0], eta[1] = eta[1], eta_next
            self.eta = eta
            self.Fh = self.K * np.power(np.abs(eta_next - y_next[self.i0]), self.p)

        # シフト
        y[0, :], y[1, :], y[2, :] = y[1, :], y[2, :], y_next
        self.y = y

        # 速度の記録
        self.u_rec.pop(0)
        self.u_rec.append((y[2, self.mic_idx] - y[1, self.mic_idx]) / dt)

        self.t += dt

    def get_displacement(self):
        return self.y[2, 1:-1]

    def get_acc(self):
        # y[n], y[n-1], y[n-2] から有限差分近似
        y = self.y
        dt2 = self.dt**2
        # 中央の時刻インデックス(2)が最新
        acc = (y[2, 1:-1] - 2 * y[1, 1:-1] + y[0, 1:-1]) / dt2
        return acc

    def display_realtime(self, sim_per_frame=1):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

        (line1,) = ax1.plot([], [], "r-")
        (line2,) = ax2.plot([], [], "b-")

        plt.subplots_adjust(hspace=0.3)

        time_text = fig.text(0.02, 0.95, "", fontsize=12)

        X = np.linspace(0, self.L, self.N + 1)

        def init():
            # ある時刻での弦の各位置での変位
            ax1.set_xlim(0, self.L)
            ax1.set_ylim(-self.display_amp, self.display_amp)

            sf = ScalarFormatter()
            sf.set_scientific(True)
            sf.set_powerlimits((-0.00001, 0.00001))
            ax1.xaxis.set_major_formatter(sf)
            ax1.yaxis.set_major_formatter(sf)
            ax2.set_autoscale_on(False)
            ax1.set_xlabel("Position [m]")
            ax1.set_ylabel("Amplitude [m]")
            T = np.linspace(self.t - self.dt * self.u_rec_num, self.t, self.u_rec_num)

            # 速度の記録
            ax2.set_xlim(T[0], T[-1])
            ax2.set_ylim(-self.display_u_amp, self.display_u_amp)
            ax2.xaxis.set_major_formatter(sf)
            ax2.yaxis.set_major_formatter(sf)
            ax2.set_autoscale_on(False)
            ax2.set_xlabel("Time [s]")
            ax2.set_ylabel("Velocity [m/s]")

            time_text.set_text(f"t = {self.t:.4f} s")
            return (line1, line2, time_text)

        def update(frame):
            for _ in range(sim_per_frame):
                self.step()
            T = np.linspace(self.t - self.dt * self.u_rec_num, self.t, self.u_rec_num)
            line1.set_data(X, self.get_displacement())
            line2.set_data(T, self.u_rec)
            ax2.set_xlim(T[0], T[-1])

            time_text.set_text(f"t = {self.t:.4f} s")
            return (line1, line2, time_text)

        ani = animation.FuncAnimation(
            fig, update, init_func=init, frames=200, interval=1, blit=False
        )
        plt.show()

    def record_audio(self, max_t=1, fs=44100, output="piano_string_audio.wav"):
        # 弦から直接音を収録
        iter_num = int(max_t / self.dt)
        mic_pressure = np.zeros(iter_num, dtype=float)
        for i in tqdm(range(iter_num)):
            self.step()
            mic_pressure[i] = self.y[2, self.mic_idx]

        mic_pressure = resample(mic_pressure, int(max_t * fs))

        mic_pressure /= np.max(np.abs(mic_pressure) + 1e-12)
        wav_data = np.int16(mic_pressure * 32767)
        write(output, fs, wav_data)

    def record_velocity(self, max_t=1, fs=44100, output="piano_string_velocity.wav"):
        # 弦の速度を収録
        iter_num = int(max_t / self.dt)
        velocity = np.zeros(iter_num, dtype=float)
        for i in tqdm(range(iter_num)):
            self.step()
            velocity[i] = self.u_rec[-1]

        velocity = resample(velocity, int(max_t * fs))

        velocity /= np.max(np.abs(velocity) + 1e-12)
        wav_data = np.int16(velocity * 32767)
        write(output, fs, wav_data)


class Room:
    """
    d次元グリッド上で波動方程式を有限差分で解く
    """

    def __init__(self, config):
        with open(config, "r") as f:
            params = json.load(f)
        self.__dict__.update(params)

        self.L = np.array(self.L, dtype=float)
        self.N = (self.L / self.dx).astype(int)
        self.dt = 1 / self.fe

        self.mic_pos = np.array(self.mic_pos, dtype=float)
        self.mic_idx = tuple((self.mic_pos / self.dx).astype(int))

        shape = (2, *(self.N + 1))
        self.p = np.zeros(shape, dtype=float)

    def step(self, source=None):
        p = self.p
        dt = self.dt
        dx = self.dx
        c2 = (self.c * dt / dx) ** 2
        gamma = self.gamma

        if source is None:
            S = 0.0
        else:
            S = source

        # 近傍セルとの和
        neighbors_sum = np.zeros_like(p[1])
        for d in range(p[1].ndim):
            neighbors_sum += np.roll(p[1], 1, axis=d) + np.roll(p[1], -1, axis=d)

        p_next = (
            2 * p[1]
            - p[0]
            - 2 * gamma * dt * (p[1] - p[0])
            + c2 * (neighbors_sum - 2 * p[1].ndim * p[1])
            + dt**2 * S
        )
        # 境界をゼロに (簡易境界条件)
        for d in range(1, p_next.ndim):
            p_next[
                (slice(None),) * (d) + (0,) + (slice(None),) * (p_next.ndim - d - 1)
            ] = 0
            p_next[
                (slice(None),) * (d) + (-1,) + (slice(None),) * (p_next.ndim - d - 1)
            ] = 0

        p[0], p[1] = p[1], p_next
        self.p = p

    def get_mic_pressure(self):
        return self.p[1][self.mic_idx]


class PianoInRoomSimulation:
    """
    弦シミュレーション + 部屋シミュレーションをまとめて制御する
    """

    def __init__(self, piano_string: PianoString, room: Room, dt=1e-4):
        self.string = piano_string
        self.room = room

        # 将来的には独立時間刻みを設定できるようにする予定
        self.dt = dt
        self.string.dt = dt
        self.room.dt = dt

        # 弦の位置は一旦roomのconfigに入れておく
        self.string_pos = self.room.string_pos
        # どの軸に弦を配置するか (0:x, 1:y, 2:z, ...)
        axis = 0
        self.string_axis = axis
        self.string_grid = [int(p / room.dx) for p in self.string_pos]
        self.string_grid[axis] = slice(
            int((self.string_pos[axis] - piano_string.L / 2) / room.dx),
            int((self.string_pos[axis] + piano_string.L / 2) / room.dx),
        )
        self.string_grid = tuple(self.string_grid)

        # 力の伝達係数(音圧を見て調整)
        # これも一旦roomのconfigに入れておく
        self.c_str_to_room = self.room.c_str_to_room

        self.room_source = np.zeros(room.N + 1, dtype=float)
        # 空気は弦へ影響を与えないとする
        # self.string_source = np.zeros(piano_string.N + 3, dtype=float)

    def step(self):
        # 弦を更新する
        self.string.step()
        # 部屋を更新する
        self.room.step(source=self.room_source)

        # 部屋のsource項を計算する
        acc = self.string.get_acc()
        string_slice = self.string_grid[self.string_axis]
        acc = resample(acc, string_slice.stop - string_slice.start)
        shape = (*(self.room.N + 1),)
        source = np.zeros(shape, dtype=float)
        source[self.string_grid] = self.c_str_to_room * acc
        self.room_source = source

    def display_realtime(self, sim_per_frame=1):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        string = self.string
        room = self.room

        # 1D string
        (line,) = ax1.plot([], [], "r-")
        X = np.linspace(0, string.L, string.N + 1)
        ax1.set_xlim(0, string.L)
        ax1.set_ylim(-string.display_amp, string.display_amp)

        # 2D room
        grid = list(self.string_grid)
        # 弦の軸以外でどの軸を表示するか
        axis = (self.string_axis + 1) % room.L.shape[0]
        grid[self.string_axis] = slice(None)
        grid[axis] = slice(None)
        grid = tuple(grid)

        im = ax2.imshow(
            room.p[1][grid],
            cmap="viridis",
            vmin=-room.display_amp,
            vmax=room.display_amp,
            origin="lower",
            extent=[0, 1, 0, 1],
        )
        print(grid)

        def init():
            line.set_data([], [])
            im.set_data(np.zeros_like(room.p[1][grid]))
            return line, im

        def update(frame):
            for _ in range(sim_per_frame):
                self.step()
            # Update 1D string properly
            line.set_data(X, self.string.get_displacement())
            # Update 2D room
            im.set_data(room.p[1][grid])
            return line, im

        ani = animation.FuncAnimation(
            fig, update, init_func=init, frames=200, interval=1, blit=False
        )
        plt.show()

    def record_audio(self, duration_sec, fs=44100, wavfile="piano_in_room.wav"):
        steps = int(duration_sec / self.dt)
        mic_pressure = np.zeros(steps, dtype=float)
        for i in tqdm(range(steps)):
            self.step()
            mic_pressure[i] = self.room.get_mic_pressure()

        if fs != (1 / self.dt):
            out_len = int(len(mic_pressure) * fs * self.dt)
            mic_pressure = resample(mic_pressure, out_len)

        mic_pressure /= np.max(np.abs(mic_pressure))
        wav_data = np.int16(mic_pressure * 32767)
        write(wavfile, fs, wav_data)


if __name__ == "__main__":
    piano_string = PianoString("params/piano_string_c4.json")
    # CFL条件の確認
    print(f"CFL condition: {piano_string.c * piano_string.dt / piano_string.dx}")
    room = Room("params/room_2d.json")
    # CFL条件の確認
    print(f"CFL condition: {room.c * room.dt / room.dx}")
    # fe = 32e3
    fe = max(piano_string.fe, room.fe)
    dt = 1 / fe
    sim = PianoInRoomSimulation(piano_string, room, dt=dt)
    # sim.display_realtime()
    sim.record_audio(2, fs=44100, wavfile="piano_in_room.wav")
    print("Done.")
