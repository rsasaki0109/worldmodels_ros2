"""A *playable* episodic world model: steer, and it imagines the road ahead.

This is the interactive face of the same learning-free machinery used by the
planner and the counterfactual demo. Instead of asking the model "what if I
steer left vs right?" once, you sit in the loop: every frame you choose a
steering action, the world model predicts the next latent (marching forward
through real driving experience), and decodes it to a real frame. You are
literally *driving inside the model* -- no simulator, no rendering engine, no
dynamics training. The whole game runs in the frozen encoder's latent space.

This is the retrieval analogue of "playable world models" (DIAMOND / Oasis /
Navigation World Models): there the next frame is *generated* by a diffusion /
autoregressive model trained for the purpose; here it is *retrieved* from a
memory of what really happened, conditioned on your action. No pixels are
hallucinated, and nothing is trained -- the trade-off is honest.

``DriveSim`` is plain numpy, ROS-free and torch-free, so it is unit-tested on CI
with a toy rotation world and reused by the interactive front-end (play_drive.py)
on real public driving video.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .planning import RetrievalDynamics, cosine_distance


@dataclass
class DriveSim:
    """An interactive episodic world model you can steer through.

    Built from real driving experience:
      * ``dynamics``        action-conditioned :class:`RetrievalDynamics`
                            ``(latent_t, steering_t) -> latent_{t+1}``
      * ``memory_latents``  (N, D) latents to decode against (nearest neighbour)
      * ``memory_frames``   the N real frames those latents belong to
      * ``start_latent``    where the drive begins
      * ``start_index``     its row in the transition memory (for forward bias)

    Call :meth:`reset` to go back to the start, then :meth:`step` with a steering
    value each tick. The sim marches *forward* through the experience (so the
    drive keeps moving) while your steering bends which future it retrieves.
    """

    dynamics: RetrievalDynamics
    memory_latents: np.ndarray
    memory_frames: list
    start_latent: np.ndarray
    start_index: int = 0
    forward_bias: bool = False
    decoder: object = None          # optional LatentDecoder: generate, don't retrieve

    latent: np.ndarray = field(init=False, default=None)
    cursor: int = field(init=False, default=-1)
    t: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.memory_latents = np.asarray(self.memory_latents, dtype=np.float32)
        if len(self.memory_frames) != self.memory_latents.shape[0]:
            raise ValueError("memory_frames and memory_latents must have equal length")
        # Pre-normalise the decode memory once so each frame() is a single matmul.
        self._mem_n = self.memory_latents / (
            np.linalg.norm(self.memory_latents, axis=-1, keepdims=True) + 1e-8
        )
        self.reset()

    def reset(self):
        """Return to the start latent/frame; returns the start frame."""
        self.latent = np.asarray(self.start_latent, dtype=np.float32).ravel()
        self.cursor = int(self.start_index)
        self.t = 0
        return self.frame()

    def frame(self):
        """Render the current latent: *generate* it with the decoder if one is
        attached, otherwise retrieve the nearest real frame."""
        if self.decoder is not None:
            return self.decoder.decode(self.latent)
        q = self.latent / (np.linalg.norm(self.latent) + 1e-8)
        return self.memory_frames[int(np.argmax(self._mem_n @ q))]

    def step(self, steering: float):
        """Advance one tick under ``steering``; returns (frame, latent).

        With ``forward_bias`` the sim marches forward in time (smooth motion, but
        locked to one episode's real steering, so commanded steering barely
        changes the route). Without it (the default) commanded steering genuinely
        selects the future -- left vs right diverge strongly -- at the cost of
        occasionally lingering on a remembered state. See the module docstring.
        """
        a = np.array([float(steering)], dtype=np.float32)
        after = self.cursor if self.forward_bias else -1
        self.latent, idx = self.dynamics.step_with_index(self.latent, a, after)
        self.cursor = idx
        self.t += 1
        return self.frame(), self.latent

    def drive(self, steering_sequence):
        """Roll a whole steering sequence from the start; returns the frames."""
        self.reset()
        frames = [self.frame()]
        for s in steering_sequence:
            f, _ = self.step(s)
            frames.append(f)
        return frames


def load_world(npz_path: str, dynamics: str = "auto",
               decoder_path: str = None) -> DriveSim:
    """Build a :class:`DriveSim` from a ``.npz`` produced by build_drive_world.py.

    Expected arrays: ``latents`` (N,D), ``actions`` (N,1) steering,
    ``next_latents`` (N,D), ``frames`` (N,H,W,3) uint8, and optional scalars
    ``start_index`` / ``action_weight`` / ``k``. If the file also carries a
    trained head (``n_layers`` etc.) it is used by default -- that is what makes
    the world smoothly *playable*.

    ``dynamics``: ``"learned"`` (the MLP head), ``"retrieval"`` (learning-free),
    or ``"auto"`` (learned if present, else retrieval).
    """
    z = np.load(npz_path, allow_pickle=False)
    lat, act, nxt = z["latents"], z["actions"], z["next_latents"]
    frames = list(z["frames"])
    start = int(z["start_index"]) if "start_index" in z else 0
    has_head = "n_layers" in z
    use_learned = dynamics == "learned" or (dynamics == "auto" and has_head)
    if use_learned:
        from .dynamics import MLPDynamics
        dyn = MLPDynamics.load(z)
        forward_bias = False               # the head moves on its own
    else:
        k = int(z["k"]) if "k" in z else 6
        aw = float(z["action_weight"]) if "action_weight" in z else 0.4
        dyn = RetrievalDynamics(lat, act, nxt, k=k, action_weight=aw)
        forward_bias = True                # retrieval needs the forward march
    decoder = None
    if decoder_path is not None:
        from .decoder import LatentDecoder
        decoder = LatentDecoder.load(decoder_path)
    # Decode against the *arrival* latents (next_latents) so a decoded latent
    # shows where you end up, matching the counterfactual demo.
    return DriveSim(dyn, nxt, frames, start_latent=nxt[start],
                    start_index=start, forward_bias=forward_bias, decoder=decoder)
