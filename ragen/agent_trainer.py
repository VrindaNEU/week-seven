"""
Multi-turn agent trainer - extends APOTrainer for RAGEN
DOES NOT modify original TinyZero code!
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional
from collections import deque

# Import YOUR working TinyZero trainer
from tinyzero.apo_trainer import APOTrainer
from tinyzero.rewards import compute_reward


class _SimpleVStarCache:
    """Lightweight in-memory cache keyed by (prompt, apo-signature)."""
    def __init__(self):
        self._store = {}
    def _key(self, prompt, config):
        apo = config.get('apo', {})
        sig = (
            apo.get('beta', 0.5),
            apo.get('v_star_samples', 4),
            apo.get('adaptive_vstar', False),
        )
        return (prompt, sig)
    def get(self, prompt, config):
        return self._store.get(self._key(prompt, config))
    def save(self, prompt, config, data):
        self._store[self._key(prompt, config)] = data


class MultiTurnAPOTrainer(APOTrainer):
    """
    Multi-turn extension of APOTrainer.

    TinyZero:  prompt → single response → reward
    RAGEN:     prompt → [turn1, turn2, ..., turnN] → trajectory_reward
    """

    def __init__(self, policy_model, reference_model, config: Dict, environment):
        # Pull in all stable fixes from parent (weights, KL, masking, etc.)
        super().__init__(policy_model, reference_model, config)

        # Environment
        self.environment = environment
        self.max_turns = config.get('environment', {}).get('max_turns', 10)

        # Safe defaults for multi-turn extras
        apo_cfg = config.get('apo', {})
        self.weighting_scheme = apo_cfg.get('weighting_scheme', 'exp')  # 'exp' | 'adv' | 'shifted_advantage'
        self.adaptive_vstar = apo_cfg.get('adaptive_vstar', False)
        self.clip_grad_norm = apo_cfg.get('clip_grad_norm', 0.5)

        # Optional adaptive gradient clipping (OFF by default)
        self.adaptive_clip = apo_cfg.get('adaptive_clip', False)
        self._grad_hist = deque(maxlen=100)

        # Some generators don't support min_new_tokens; guard it
        samp_cfg = config.get('sampling', {})
        self.use_min_new_tokens = samp_cfg.get('use_min_new_tokens', False)
        self.min_new_tokens = samp_cfg.get('min_new_tokens', 10)

        # Simple in-memory V* cache
        self.vstar_cache = _SimpleVStarCache()

        print(f"✓ Multi-turn trainer initialized (max_turns={self.max_turns})")

    # ---------- small robustness helpers ----------

    def _safe_exp_weights(self, advantages: torch.Tensor) -> torch.Tensor:
        """
        Stable exp((r - V*)/beta) with clipping/capping.
        Returns weights normalized to mean=1.
        """
        beta = self.beta + 1e-8
        x = (advantages / beta).clamp_(-20.0, 20.0)  # prevent overflow
        w = torch.exp(x)
        w = torch.clamp(w, max=50.0)                # avoid a few samples dominating
        return w / (w.mean().clamp_min(1e-6))

    def _grad_global_norm(self) -> float:
        total_sq = 0.0
        for p in self.policy.parameters():
            if p.grad is not None:
                g = p.grad.data
                total_sq += float(g.norm(2).item() ** 2)
        return float(total_sq ** 0.5)

    def _maybe_adapt_clip(self, unclipped_norm: float):
        """
        Optionally adapt clip_grad_norm using recent gradient norms.
        Sets clip to ~2x median of recent norms.
        """
        if not self.adaptive_clip:
            return
        self._grad_hist.append(unclipped_norm)
        # Update every 10 steps after we have a decent buffer
        if self.step >= 100 and (self.step % 10 == 0) and len(self._grad_hist) >= 20:
            med = float(np.median(self._grad_hist))
            new_clip = max(0.1, 2.0 * med)
            # Smooth changes a bit
            self.clip_grad_norm = 0.8 * self.clip_grad_norm + 0.2 * new_clip
            print(f"  ↻ Adaptive clip_grad_norm → {self.clip_grad_norm:.3f} (median={med:.3f})")

    # ---------- token-level KL on completion tokens only ----------

    def _compute_kl_per_example(self, pi_logits: torch.Tensor, ref_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        KL(pi || ref) computed on completion tokens only. Returns per-example KL.
        pi_logits/ref_logits: [B, T, V]
        labels: [B, T] with -100 mask for prompt tokens
        """
        with torch.no_grad():
            token_mask = (labels != -100).float()  # [B, T]
        logp_pi = F.log_softmax(pi_logits, dim=-1)
        logp_ref = F.log_softmax(ref_logits, dim=-1)
        gather_idx = labels.clone()
        gather_idx[gather_idx == -100] = 0  # safe index
        lp_pi = logp_pi.gather(-1, gather_idx.unsqueeze(-1)).squeeze(-1)   # [B, T]
        lp_ref = logp_ref.gather(-1, gather_idx.unsqueeze(-1)).squeeze(-1) # [B, T]
        kl_tokens = (lp_pi - lp_ref) * token_mask                           # [B, T]
        per_ex_kl = (kl_tokens.sum(dim=1) / token_mask.sum(dim=1).clamp_min(1.0))  # [B]
        return per_ex_kl

    # ---------- state→text adapter ----------

    def _state_to_text(self, state) -> str:
        """Convert env state to a text prompt for the policy."""
        if hasattr(self.environment, "render_text"):
            return self.environment.render_text(state)
        return str(state)

    # ---------- multi-turn rollout ----------

    def _rollout_trajectory(self, prompt: str, task_data: Dict, max_turns: Optional[int] = None) -> Optional[Dict]:
        """
        Collect a multi-turn trajectory with robust error handling.
        Returns None if rollout completely fails.
        """
        if max_turns is None:
            max_turns = self.max_turns

        try:
            # Reset env with task
            task = dict(task_data)  # avoid mutating caller
            task['prompt'] = prompt
            state = self.environment.reset(task)
            
            # Validate initial state
            if state is None or not state:
                print(f"⚠️  Invalid initial state from reset: {state}")
                return None

            actions, rewards, states = [], [], [state]
            done = False

            for turn in range(max_turns):
                try:
                    obs_text = self._state_to_text(state)
                    
                    # Validate observation text
                    if not obs_text or not isinstance(obs_text, str):
                        print(f"  Invalid observation at turn {turn}: {obs_text}")
                        break
                    
                    gen_kwargs = dict(
                        max_length=max(16, self.min_new_tokens if self.use_min_new_tokens else 32),
                        temperature=self.temperature,
                        do_sample=True,
                        top_p=self.top_p,
                        top_k=self.top_k,
                        repetition_penalty=1.3,
                    )
                    if self.use_min_new_tokens:
                        gen_kwargs["min_new_tokens"] = self.min_new_tokens

                    # Generate action
                    generated = self.policy.generate([obs_text], **gen_kwargs)
                    if not generated or not generated[0]:
                        print(f"  Empty generation at turn {turn}")
                        break
                    action = generated[0]

                    # Sanitize model output for environment compatibility
                    from ragen.action_sanitizer import sanitize
                    fallback_query = task.get("prompt", "")
                    action = sanitize(action, fallback_query)


                    # ADD THIS DEBUG LOGGING:
                    if turn == 0 and self.step <= 5:  # Log first action of first 5 steps
                        print(f"   Step {self.step}, Turn {turn}:")
                        print(f"     Obs: {obs_text[:80]}...")
                        print(f"     Action: '{action}'")
                    
                    # Execute step
                    next_state, reward, done, info = self.environment.step(action)


                    # ADD THIS TOO:
                    if turn == 0 and self.step <= 5:
                        print(f"     Reward: {reward}, Done: {done}")
                    
                    # Validate step results
                    if next_state is None:
                        print(f"  None state returned at turn {turn}")
                        # Use the error message as state if available
                        next_state = info.get('error', 'Error occurred')
                    
                    actions.append(action)
                    rewards.append(float(reward))
                    states.append(next_state)
                    state = next_state
                    
                    if done:
                        break
                        
                except Exception as e:
                    print(f"⚠️  Error during turn {turn}: {e}")
                    # Don't break immediately - record what we have so far
                    break

            # Validate trajectory before returning
            if not actions:
                print(f"  No actions collected in trajectory")
                return None
            
            total_reward = float(np.sum(rewards)) if rewards else 0.0
            full_text = self._concat_trajectory(actions)
            
            # Ensure we have valid text
            if not full_text:
                print(f"  Empty full_text after concatenation")
                full_text = " "  # Minimum valid text

            # ADD THIS LOGGING:
            if self.step <= 10:  # First 10 steps only
                print(f"     Trajectory {self.step}: {len(actions)} actions")
                print(f"     First 3 actions: {[str(a)[:50] for a in actions[:3]]}")
                print(f"     Rewards: {rewards}")
            
            return {
                'states': states,
                'actions': actions,
                'rewards': rewards,
                'total_reward': total_reward,
                'full_text': full_text,
                'num_turns': len(actions),
                'done': done
            }
            
        except Exception as e:
            print(f"  Rollout completely failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _concat_trajectory(self, actions: List[str]) -> str:
        """
        Concatenate actions into a single completion string.
        (Prompt will be provided separately and masked.)
        """
        if not actions:
            return ""
        return "\n".join(a.strip() for a in actions if a is not None)

    # ---------- V* for multi-turn ----------

    def compute_V_star_multiturn(self, prompts: List[str], problems: List[Dict]) -> np.ndarray:
        # Fast warm-up heuristic (cheap & stable early on)
        if self.step < 20:
            print(f"  ⚡ Using FAST V* estimation (step {self.step})...")
            return np.full(len(prompts), 0.3, dtype=np.float32)

        V_star_values = []
        cache_hits = 0

        # Adaptive sampling
        if self.adaptive_vstar:
            if self.step < 30:
                num_samples = 2
            elif self.step < 70:
                num_samples = 1
            else:
                num_samples = 1
        else:
            num_samples = min(self.v_star_samples, 2)

        print(f"  Computing V* for {len(prompts)} prompts ({num_samples} samples each)...")

        # Decide which prompts need compute
        to_compute, indices = [], []
        for i, p in enumerate(prompts):
            cached = self.vstar_cache.get(p, self.config)
            if cached is not None:
                V_star_values.append(cached['v_star'])
                cache_hits += 1
            else:
                V_star_values.append(None)
                to_compute.append(p)
                indices.append(i)

        # Compute for uncached prompts
        for p, i in zip(to_compute, indices):
            problem = problems[i]
            sample_rewards = []
            for _ in range(num_samples):
                try:
                    traj = self._rollout_trajectory(p, problem, max_turns=min(self.max_turns, 5))
                    if traj is not None:
                        sample_rewards.append(traj['total_reward'])
                except Exception as e:
                    print(f"    Warning: V* sampling failed: {e}")
            if not sample_rewards:
                V_star = 0.3  # conservative fallback
            else:
                rewards = np.array(sample_rewards, dtype=np.float32)
                if self.beta > 0:
                    max_r = rewards.max()
                    # numeric stability for exponent
                    exp_terms = np.exp(np.clip((rewards - max_r) / self.beta, -20.0, 20.0))
                    V_star = float(max_r + self.beta * np.log(np.mean(exp_terms)))
                else:
                    V_star = float(rewards.max())

            self.vstar_cache.save(p, self.config, {
                'v_star': V_star,
                'rewards': sample_rewards,
                'num_samples': len(sample_rewards),
            })
            V_star_values[i] = V_star

        if cache_hits:
            hit_rate = 100.0 * cache_hits / len(prompts)
            print(f"  ✓ V* cache: {cache_hits}/{len(prompts)} hits ({hit_rate:.1f}%)")

        return np.array(V_star_values, dtype=np.float32)

    # ---------- training ----------

    def train_step_multiturn(self, batch: List[Dict]) -> tuple:
        """
        Multi-turn training step - FIXED to not train on dummy data.
        """
        prompts = [item.get('instruction', item.get('prompt', '')) for item in batch]
        device = self.policy.model.device

        try:
            self.policy.train()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 1) Collect trajectories - ONLY keep successful ones
            trajectories = []
            successful_indices = []
            rollout_failures = 0
            
            for i, (p, prob) in enumerate(zip(prompts, batch)):
                traj = self._rollout_trajectory(p, prob, self.max_turns)
                if traj is not None:
                    trajectories.append(traj)
                    successful_indices.append(i)
                else:
                    rollout_failures += 1
            
            # Skip step if too many failures
            if len(trajectories) < max(1, len(batch) * 0.3):
                print(f"  ⚠️  Too many failures ({rollout_failures}/{len(batch)}) - skipping")
                return 0.0, {
                    'loss': 0.0, 'avg_reward': 0.0, 'avg_advantage': 0.0,
                    'avg_v_star': 0.0, 'avg_kl_penalty': 0.0,
                    'avg_turns': 0.0, 'success_rate': 0.0,
                    'weight_mean': 1.0, 'weight_std': 0.0,
                    'grad_global_norm_unclipped': 0.0,
                    'rollout_failures': rollout_failures,
                }
            
            if rollout_failures > 0:
                print(f"  ⚠️  {rollout_failures}/{len(batch)} rollouts failed")
            
            # 2) Align everything to successful rollouts
            prompts = [prompts[i] for i in successful_indices]
            batch = [batch[i] for i in successful_indices]
            
            # 3) Compute V* only for successful prompts
            V_star_np = self.compute_V_star_multiturn(prompts, problems=batch)
            V_star_t = torch.tensor(V_star_np, dtype=torch.float32, device=device)

            generated_texts = [t['full_text'] for t in trajectories]

            # 4) Compute rewards
            computed_rewards = []
            for traj, prob in zip(trajectories, batch):
                reward = compute_reward(traj, prob)
                computed_rewards.append(reward)

            rewards_t = torch.tensor(computed_rewards, dtype=torch.float32, device=device)

            if self.step % 5 == 0:
                print(f"  📊 Rewards: {computed_rewards}")

            # 5) Advantages & weights
            advantages = rewards_t - V_star_t
            if len(advantages) > 1:
                adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
            else:
                adv_norm = advantages - advantages.mean()
            adv_norm = adv_norm.clamp(-self.adv_clip, self.adv_clip).detach()

            if self.weighting_scheme == 'exp':
                weights = self._safe_exp_weights(advantages).detach()
            elif self.weighting_scheme == 'shifted_advantage':
                weights = (adv_norm + self.adv_clip).detach()
                weights = weights / (weights.mean().clamp_min(1e-6))
            else:
                weights = (adv_norm + 1.0).clamp(min=0.1, max=5.0).detach()
                weights = weights / (weights.mean().clamp_min(1e-6))

            # 6) Tokenization
            enc_prompts = self.policy.tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True,
                max_length=self.sft_max_length
            )
            enc_comps = self.policy.tokenizer(
                generated_texts, return_tensors="pt", padding=True, truncation=True,
                max_length=self.sft_max_length
            )
            enc_prompts = {k: v.to(device) for k, v in enc_prompts.items()}
            enc_comps = {k: v.to(device) for k, v in enc_comps.items()}

            pad_id = self.policy.tokenizer.pad_token_id or self.policy.tokenizer.eos_token_id

            input_ids, attention_mask, labels = self._build_concat_with_labels(
                enc_prompts["input_ids"], enc_comps["input_ids"], pad_id
            )
            input_ids = input_ids[:, :self.sft_max_length]
            attention_mask = attention_mask[:, :self.sft_max_length]
            labels = labels[:, :self.sft_max_length]

            # 7) Forward pass
            outputs = self.policy.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None
            )
            per_ex_ce = self._per_example_ce_loss(outputs.logits, labels)

            # 8) KL term
            kl_term = torch.zeros_like(per_ex_ce)
            if self.kl_coef and self.kl_coef > 0.0 and hasattr(self.ref_model, "model"):
                with torch.no_grad():
                    ref_out = self.ref_model.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                kl_per_ex = self._compute_kl_per_example(outputs.logits, ref_out.logits, labels)
                kl_term = self.kl_coef * kl_per_ex
                del ref_out
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            per_ex_total = per_ex_ce + kl_term

            # 9) Weighted loss
            if torch.isnan(weights).any() or torch.isinf(weights).any():
                print("Warning: NaN/Inf weights")
                weights = torch.ones_like(per_ex_total)
            weights = weights.to(device)
            loss = (per_ex_total * weights).mean()

            # 10) Backprop
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            unclipped_norm = self._grad_global_norm()
            self._maybe_adapt_clip(unclipped_norm)
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.clip_grad_norm)
            self.optimizer.step()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 11) Metrics
            self.step += 1
            loss_value = float(loss.item())
            avg_reward = float(rewards_t.mean().item())
            avg_advantage = float(advantages.mean().item())
            avg_v_star = float(V_star_t.mean().item())
            avg_kl = float(kl_term.mean().item())
            avg_turns = float(np.mean([t['num_turns'] for t in trajectories]))
            success_rate = float(np.mean([1.0 if t['total_reward'] > 0.5 else 0.0 for t in trajectories]))

            w_mean = float(weights.mean().item())
            w_std  = float(weights.std().item())

            if self.step % self.config.get('logging', {}).get('log_every', 5) == 0:
                print(
                    f"Step {self.step}: Loss={loss_value:.4f}, "
                    f"Reward={avg_reward:.3f}, Success={success_rate:.2%}"
                )

            stats = {
                'loss': loss_value,
                'avg_reward': avg_reward,
                'avg_advantage': avg_advantage,
                'avg_v_star': avg_v_star,
                'avg_kl_penalty': avg_kl,
                'avg_turns': avg_turns,
                'success_rate': success_rate,
                'weight_mean': w_mean,
                'weight_std': w_std,
                'grad_global_norm_unclipped': unclipped_norm,
                'rollout_failures': rollout_failures,
            }
            return loss_value, stats

        except Exception as e:
            print(f"\n--- Error in train_step_multiturn ---")
            print(f"{type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return 0.0, {
                'loss': 0.0, 'avg_reward': 0.0, 'avg_advantage': 0.0,
                'avg_v_star': 0.0, 'avg_kl_penalty': 0.0,
                'avg_turns': 0.0, 'success_rate': 0.0,
                'weight_mean': 1.0, 'weight_std': 0.0,
                'grad_global_norm_unclipped': 0.0,
                'rollout_failures': len(batch),
            }