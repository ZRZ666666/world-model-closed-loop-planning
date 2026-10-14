"""Complete paper-based Sandwich-Residuals port for Temporal visual8.

Source: arXiv:2609.21740v1 III-A Eqs4-6, III-B Eqs7-13. This is local
implementation, not author code or reproduction of their CEM benchmark.
Four branches: two shared patch MLP8->64->8, action20->10, proprio10->10.
The Temporal mean18-channel observation loss is an explicit architecture port.

Local choices not resolved by author code: persistent Adam across cycles;
PyTorch-default first MLP Linear, final weight Normal(std=1e-4), final bias0.
Linear residuals start zero. Initialization is near-identity, NOT exact.
"""
from contextlib import contextmanager
from types import MethodType
import torch
from torch import nn


class _Installation:
    """Plain holder: never register the backbone under the residual Module."""
    def __init__(self,core,wm):
        assert not hasattr(wm,'_wm29_sandwich')
        assert wm.concat_dim==1 and wm.action_dim==wm.proprio_dim==10
        assert wm.num_action_repeat==wm.num_proprio_repeat==1 and wm.num_hist==3
        assert wm.encoder.emb_dim==8
        self.core,self.model=core,wm
        self.original_encode_obs,self.original_predict=wm.encode_obs,wm.predict
        self.previous_encode=wm.__dict__.get('encode_obs')
        self.previous_predict=wm.__dict__.get('predict')
        for p in wm.parameters(): p.requires_grad_(False);p.grad=None
        wm.eval()
        anchor=next(wm.parameters())
        assert anchor.device==next(core.parameters()).device and anchor.dtype==torch.float32
        binding=self
        def encode_obs(self,obs):
            raw=binding.original_encode_obs(obs)
            if binding.core._disabled_depth:return raw
            assert raw['visual'].shape[-2:]==(196,8)
            return {**raw,'visual':raw['visual']+binding.core.visual_in(raw['visual'])}
        def predict(self,z):
            if binding.core._disabled_depth:return binding.original_predict(z)
            x=binding.core.correct_predict_input(z)
            return binding.core.correct_predict_output(binding.original_predict(x))
        wm.encode_obs=MethodType(encode_obs,wm)
        wm.predict=MethodType(predict,wm)
        wm._wm29_sandwich=self

    def remove(self):
        for name,prior in [('encode_obs',self.previous_encode),('predict',self.previous_predict)]:
            if prior is None:delattr(self.model,name)
            else:setattr(self.model,name,prior)
        delattr(self.model,'_wm29_sandwich')
        # Backbone remains frozen/eval; restoration is caller policy.


class SandwichCore(nn.Module):
    def __init__(self,device,optimizer_policy='persistent'):
        super().__init__()
        assert optimizer_policy=='persistent'
        self.optimizer_policy=optimizer_policy
        self.visual_in=nn.Sequential(nn.Linear(8,64),nn.GELU(),nn.Linear(64,8))
        self.visual_out=nn.Sequential(nn.Linear(8,64),nn.GELU(),nn.Linear(64,8))
        self.action_in=nn.Linear(20,10)
        self.proprio_out=nn.Linear(10,10)
        for m in (self.action_in,self.proprio_out):
            nn.init.zeros_(m.weight);nn.init.zeros_(m.bias)
        for m in (self.visual_in,self.visual_out):
            nn.init.normal_(m[-1].weight,mean=0,std=1e-4);nn.init.zeros_(m[-1].bias)
        self.to(device=device,dtype=torch.float32)
        assert sum(p.numel() for p in self.parameters())==2512
        self._initial_state={k:v.detach().cpu().clone() for k,v in self.state_dict().items()}
        self._installation=None
        self._disabled_depth=0
        self.buffer=[]
        self.fit_calls=self.optimizer_steps=self.training_backward_calls=0
        self.optimizer=self._new_optimizer()
        self.requires_grad_(False)

    def _new_optimizer(self):
        return torch.optim.Adam([
            {'params':list(self.action_in.parameters())+list(self.proprio_out.parameters()),'lr':.01,'group_name':'linear'},
            {'params':list(self.visual_in.parameters())+list(self.visual_out.parameters()),'lr':.0005,'group_name':'visual'}])

    def install(self,wm):
        assert self._installation is None
        self._installation=_Installation(self,wm)
        return self._installation

    @contextmanager
    def planning(self):
        previous=[p.requires_grad for p in self.parameters()]
        self.requires_grad_(False)
        try:yield
        finally:
            for p,enabled in zip(self.parameters(),previous):p.requires_grad_(enabled)

    @contextmanager
    def disabled(self):
        self._disabled_depth+=1
        try:yield
        finally:self._disabled_depth-=1

    @staticmethod
    def _tokens(z):
        assert z.ndim==4 and z.shape[-2:]==(196,28)

    def correct_predict_input(self,z):
        self._tokens(z)
        # Encoder output / preceding corrected forecast already has its visual
        # correction. Never add visual-in again on recursive predictor calls.
        v,p,a=z[...,:8],z[...,8:18],z[...,18:]
        return torch.cat((v,p,a+self.action_in(torch.cat((a,p),dim=-1))),dim=-1)

    def correct_predict_output(self,z):
        self._tokens(z)
        v,p,a=z[...,:8],z[...,8:18],z[...,18:]
        return torch.cat((v+self.visual_out(v),p+self.proprio_out(p),a),dim=-1)

    @staticmethod
    def window_loss(predicted,target):
        """Only final next observation; all corrected target features stopgrad."""
        assert predicted.ndim==target.ndim==4
        assert predicted.shape[0]==target.shape[0] and target.shape[1]==1
        assert predicted.shape[-2:]==target.shape[-2:]==(196,28)
        return (predicted[:,-1:,:,:18]-target[...,:18].detach()).square().mean()

    @staticmethod
    def _own_obs(obs):
        assert set(obs)=={'visual','proprio'}
        assert all(isinstance(x,torch.Tensor) and x.dtype==torch.float32 for x in obs.values())
        assert obs['visual'].ndim==5 and obs['visual'].shape[:3]==(1,1,3)
        assert obs['proprio'].ndim==3 and obs['proprio'].shape[:2]==(1,1)
        assert all(torch.isfinite(x).all() for x in obs.values())
        return {k:v.detach().cpu().clone() for k,v in obs.items()}

    def fit_completed(self,pair_id,start,end,current_obs,next_obs,executed_action_model):
        assert self._installation is not None and not self._disabled_depth
        assert pair_id==start//5 and end==start+5
        assert executed_action_model.shape==(1,1,10) and executed_action_model.dtype==torch.float32
        assert torch.isfinite(executed_action_model).all()
        current,nxt=self._own_obs(current_obs),self._own_obs(next_obs)
        if self.buffer:
            previous=self.buffer[-1]
            assert start==previous['end'] and pair_id==previous['pair_id']+1
            assert all(torch.equal(current[k],previous['next'][k]) for k in current)
        else:assert pair_id==start==0
        self.buffer.append(dict(pair_id=pair_id,start=start,end=end,current=current,next=nxt,
            command=executed_action_model.detach().cpu().clone()))
        self.buffer=self.buffer[-5:]
        N=len(self.buffer);device=next(self.parameters()).device
        obs={k:torch.cat([self.buffer[0]['current'][k]]+[r['next'][k] for r in self.buffer],dim=1).to(device)
             for k in ('visual','proprio')}
        actions=torch.cat([r['command'] for r in self.buffer],dim=1).to(device)
        # The target-only final action is a dummy; target loss excludes actions.
        actions=torch.cat((actions,torch.zeros_like(actions[:,:1])),dim=1)
        windows=[(length,first) for length in range(1,min(3,N)+1) for first in range(N-length+1)]
        W=len(windows)
        self.requires_grad_(True)
        self.optimizer.zero_grad(set_to_none=True)
        traces=[]
        try:
            model=self._installation.model
            assert all(not p.requires_grad for p in model.parameters())
            # Encoder weights/input observations are frozen, but visual-in
            # remains differentiable. Do not place this encode under no_grad.
            z=model.encode(obs,actions)
            self._tokens(z)
            assert z.shape[1]==N+1
            for index,(length,first) in enumerate(windows):
                context=z[:,first:first+length]
                target=z[:,first+length:first+length+1]
                predicted=model.predict(context)
                loss=self.window_loss(predicted,target)
                assert torch.isfinite(loss)
                # Accumulate equal window mean gradients, keeping only the
                # small shared input residual graph. One Adam step, not W
                # sequential updates; predictor graphs are released per window.
                (loss/W).backward(retain_graph=index<W-1)
                self.training_backward_calls+=1
                traces.append(dict(window_index=index,length=length,
                    context_pair_ids=[r['pair_id'] for r in self.buffer[first:first+length]],
                    target_after_pair=self.buffer[first+length-1]['pair_id'],
                    mse_before_update=float(loss.detach()),target_stop_gradient=True))
                del predicted,loss
            gradients={}
            for branch in ('visual_in','visual_out','action_in','proprio_out'):
                params=list(getattr(self,branch).parameters())
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
                gradients[branch]=float(torch.sqrt(sum(p.grad.detach().square().sum() for p in params)))
            assert all(p.grad is None for p in model.parameters())
            self.optimizer.step();self.optimizer_steps+=1;self.fit_calls+=1
            assert all(torch.isfinite(p).all() for p in self.parameters())
        finally:
            self.optimizer.zero_grad(set_to_none=True)
            self.requires_grad_(False)
        return dict(pair_id=pair_id,physical_start=start,physical_end=end,buffer_pair_ids=[r['pair_id'] for r in self.buffer],
            buffer_time_ranges=[[r['start'],r['end']] for r in self.buffer],parameters=2512,
            windows=W,window_trace=traces,loss_mean_before_update=sum(r['mse_before_update'] for r in traces)/W,
            gradient_norms=gradients,Adam_steps_this_fit=1,optimizer_steps_total=self.optimizer_steps,
            backward_calls_this_fit=W,backward_calls_total=self.training_backward_calls,
            optimizer_policy='persistent_local_choice',parameter_persistence='within episode',
            source='Own completed preprocessed RGB/proprio/actions only; no goal or hidden state passed.',
            operator='One predict on each actual observed context; final-next MSE, not recursive training rollout.',
            target='Whole corrected visual+original proprio target detached; input gradients retained through predictor.',
            reduction='Equal windows; mean B*196patch*18 observation channels, Temporal port choice.',
            budget=dict(encode_obs_calls=1,encoded_frames=N+1,predict_calls=W,
                predictor_context_frames=sum(length for length,first in windows),
                selected_final_next_predictions=W,returned_predict_positions=sum(length for length,first in windows),
                backward_calls=W,optimizer_steps=1),
            initial_identity=False,initialization='MLP last weight Normal std1e-4, last bias0, default first layer; linear residuals0.')

    def reset_episode(self):
        self.load_state_dict(self._initial_state)
        self.buffer=[]
        self.fit_calls=self.optimizer_steps=self.training_backward_calls=0
        self._disabled_depth=0
        self.optimizer=self._new_optimizer()
        self.requires_grad_(False)
