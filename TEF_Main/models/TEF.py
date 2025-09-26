
import torch
import torch.nn as nn
import torch.nn.functional as F
import image

# ======================== Global Configuration ========================
FUSED_NB_FEATS = 128
FUSION_METHODS = ['add', 'mul', 'cat', 'max', 'avg']
FUSION_LIST = '4-6-2-4-2-3-4-5-4-4-4-4-0-0-4'


# Feature dimensions for different views/modalities
# Uncomment and modify based on your dataset
# NB_FEATS = [128 + 10, 1000 + 10, 144 + 10, 128 + 10, 144 + 10, 73 + 10, 128 + 10, 500 + 10]
# NB_FEATS = [128, 1000, 144, 128, 144, 73, 128, 500]


# ======================== Loss Functions ========================
def KL(alpha, c):
    """
    Calculate KL divergence between Dirichlet distributions

    Args:
        alpha: Parameters of Dirichlet distribution (batch_size, num_classes)
        c: Number of classes

    Returns:
        kl: KL divergence (batch_size, 1)
    """
    beta = torch.ones((1, c)).cuda()
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)

    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)

    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)

    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def ce_loss(p, alpha, c, global_step, annealing_step):
    """
    Calculate the evidential cross-entropy loss with annealing

    Args:
        p: Ground truth labels (batch_size,)
        alpha: Predicted evidence parameters (batch_size, num_classes)
        c: Number of classes
        global_step: Current training step/epoch
        annealing_step: Total annealing steps/epochs

    Returns:
        loss: Evidential cross-entropy loss
    """
    p = p.long()
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)

    # Accuracy term
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    # Annealing coefficient for KL divergence term
    annealing_coef = min(1, global_step / annealing_step)

    # Evidence regularization term
    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c)

    return torch.mean((A + B))


# ======================== TMC Model ========================
class TMC(nn.Module):
    """
    Trusted Multi-view Classification Model
    Implements evidence-based fusion using Dempster-Shafer theory
    """

    def __init__(self, args):
        """
        Initialize TMC model

        Args:
            args: Arguments containing model configuration
        """
        super(TMC, self).__init__()
        self.args = args

        # Initialize encoders for image modalities (if applicable)
        self.rgbenc = image.ImageEncoder(args)
        self.depthenc = image.ImageEncoder(args)

        depth_last_size = args.img_hidden_sz * args.num_image_embeds
        rgb_last_size = args.img_hidden_sz * args.num_image_embeds

        # Initialize classifiers for each view
        self.clf_depth = nn.ModuleList()
        self.clf_rgb = nn.ModuleList()
        self.clf_ecape = nn.ModuleList()
        self.clf_re = nn.ModuleList()
        self.clf_fb = nn.ModuleList()
        self.clf_mf = nn.ModuleList()
        self.clf_soec = nn.ModuleList()
        self.clf_A = nn.ModuleList()
        self.clf_B = nn.ModuleList()
        self.clf_ALL = nn.ModuleList()
        self.view_models = nn.ModuleList()

        # Build classifier for ecape view
        for hidden in args.hidden:
            self.clf_ecape.append(nn.Linear(args.dims[0], hidden))
            self.clf_ecape.append(nn.BatchNorm1d(hidden))
            self.clf_ecape.append(nn.ReLU())
            # self.clf_ecape.append(nn.Dropout(args.dropout))
            depth_last_size = hidden
        self.clf_ecape.append(nn.Linear(depth_last_size, args.n_classes))

        # Build classifier for re view
        for hidden in args.hidden:
            self.clf_re.append(nn.Linear(args.dims[1], hidden))
            self.clf_re.append(nn.BatchNorm1d(hidden))
            self.clf_re.append(nn.ReLU())
            # self.clf_re.append(nn.Dropout(args.dropout))
            depth_last_size = hidden
        self.clf_re.append(nn.Linear(depth_last_size, args.n_classes))

        # Build classifier for fb view
        for hidden in args.hidden:
            self.clf_fb.append(nn.Linear(args.dims[2], hidden))
            self.clf_fb.append(nn.BatchNorm1d(hidden))
            self.clf_fb.append(nn.ReLU())
            # self.clf_fb.append(nn.Dropout(args.dropout))
            depth_last_size = hidden
        self.clf_fb.append(nn.Linear(depth_last_size, args.n_classes))

        # Build classifier for mf view
        for hidden in args.hidden:
            self.clf_mf.append(nn.Linear(args.dims[3], hidden))
            self.clf_mf.append(nn.BatchNorm1d(hidden))
            self.clf_mf.append(nn.ReLU())
            # self.clf_mf.append(nn.Dropout(args.dropout))
            depth_last_size = hidden
        self.clf_mf.append(nn.Linear(depth_last_size, args.n_classes))

        # Build classifier for soec view
        for hidden in args.hidden:
            self.clf_soec.append(nn.Linear(args.dims[4], hidden))
            self.clf_soec.append(nn.BatchNorm1d(hidden))
            self.clf_soec.append(nn.ReLU())
            # self.clf_soec.append(nn.Dropout(args.dropout))
            depth_last_size = hidden
        self.clf_soec.append(nn.Linear(depth_last_size, args.n_classes))

        # Build classifier for A view
        for hidden in args.hidden:
            self.clf_A.append(nn.Linear(args.dims[5], hidden))
            self.clf_A.append(nn.BatchNorm1d(hidden))
            self.clf_A.append(nn.ReLU())
            # self.clf_A.append(nn.Dropout(args.dropout))
            depth_last_size = hidden
        self.clf_A.append(nn.Linear(depth_last_size, args.n_classes))

        # Build classifier for B view
        for hidden in args.hidden:
            self.clf_B.append(nn.Linear(args.dims[6], hidden))
            self.clf_B.append(nn.BatchNorm1d(hidden))
            self.clf_B.append(nn.ReLU())
            # self.clf_B.append(nn.Dropout(args.dropout))
            depth_last_size = hidden
        self.clf_B.append(nn.Linear(depth_last_size, args.n_classes))

        # Optional: Build classifier for ALL view
        """
        for hidden in args.hidden:
            # self.clf_ALL.append(nn.BatchNorm1d(args.dims[7]))
            self.clf_ALL.append(nn.Linear(args.dims[7], hidden))
            # self.clf_ALL.append(nn.ReLU())
            # self.clf_B.append(nn.Dropout(args.dropout))
            depth_last_size = hidden
        self.clf_ALL.append(nn.Linear(depth_last_size, args.n_classes))
        """

        # Build depth classifier (legacy, for compatibility)
        for hidden in args.hidden:
            self.clf_depth.append(nn.Linear(depth_last_size, hidden))
            self.clf_depth.append(nn.ReLU())
            self.clf_depth.append(nn.Dropout(args.dropout))
            depth_last_size = hidden
        self.clf_depth.append(nn.Linear(depth_last_size, args.n_classes))

        # Build RGB classifier (legacy, for compatibility)
        for hidden in args.hidden:
            self.clf_rgb.append(nn.Linear(rgb_last_size, hidden))
            # self.clf_rgb.append(nn.ReLU())
            # self.clf_rgb.append(nn.Dropout(args.dropout))
            rgb_last_size = hidden
        self.clf_rgb.append(nn.Linear(rgb_last_size, args.n_classes))

    def DS_Combin_two(self, alpha1, alpha2):
        """
        Dempster-Shafer combination rule for two evidences

        Args:
            alpha1: First evidence (batch_size, num_classes)
            alpha2: Second evidence (batch_size, num_classes)

        Returns:
            alpha_a: Combined evidence (batch_size, num_classes)
        """
        # Initialize dictionaries
        alpha = dict()
        alpha[0], alpha[1] = alpha1, alpha2
        b, S, E, u = dict(), dict(), dict(), dict()

        # Calculate belief masses and uncertainties for each evidence
        for v in range(2):
            S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
            E[v] = alpha[v] - 1
            b[v] = E[v] / (S[v].expand(E[v].shape))
            u[v] = self.args.n_classes / S[v]

        # Calculate b^0 @ b^1 (outer product)
        bb = torch.bmm(b[0].view(-1, self.args.n_classes, 1), b[1].view(-1, 1, self.args.n_classes))

        # Calculate b^0 * u^1
        uv1_expand = u[1].expand(b[0].shape)
        bu = torch.mul(b[0], uv1_expand)

        # Calculate b^1 * u^0
        uv_expand = u[0].expand(b[0].shape)
        ub = torch.mul(b[1], uv_expand)

        # Calculate conflict coefficient K
        bb_sum = torch.sum(bb, dim=(1, 2), out=None)
        bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
        K = bb_sum - bb_diag

        # Calculate combined belief mass
        b_a = (torch.mul(b[0], b[1]) + bu + ub) / ((1 - K).view(-1, 1).expand(b[0].shape))

        # Calculate combined uncertainty
        u_a = torch.mul(u[0], u[1]) / ((1 - K).view(-1, 1).expand(u[0].shape))

        # Calculate new Dirichlet strength
        S_a = self.args.n_classes / u_a

        # Calculate new evidence
        e_a = torch.mul(b_a, S_a.expand(b_a.shape))
        alpha_a = e_a + 1

        return alpha_a

    def DS_Combin_seven(self, alpha1, alpha2, alpha3, alpha4, alpha5, alpha6, alpha7):
        """
        Dempster-Shafer combination rule for seven evidences
        Note: This is a placeholder implementation that currently only combines the first two

        Args:
            alpha1-alpha7: Seven evidence tensors (batch_size, num_classes)

        Returns:
            alpha_a: Combined evidence (batch_size, num_classes)
        """
        # Initialize dictionaries
        alpha = dict()
        alpha[0], alpha[1], alpha[2], alpha[3], alpha[4], alpha[5], alpha[6] = \
            alpha1, alpha2, alpha3, alpha4, alpha5, alpha6, alpha7
        b, S, E, u = dict(), dict(), dict(), dict()

        # Calculate belief masses and uncertainties for each evidence
        for v in range(7):
            S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
            E[v] = alpha[v] - 1
            b[v] = E[v] / (S[v].expand(E[v].shape))
            u[v] = self.args.n_classes / S[v]

        # Currently only combines first two evidences
        # TODO: Implement full seven-way combination
        bb = torch.bmm(b[0].view(-1, self.args.n_classes, 1), b[1].view(-1, 1, self.args.n_classes))
        uv1_expand = u[1].expand(b[0].shape)
        bu = torch.mul(b[0], uv1_expand)
        uv_expand = u[0].expand(b[0].shape)
        ub = torch.mul(b[1], uv_expand)

        bb_sum = torch.sum(bb, dim=(1, 2), out=None)
        bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
        K = bb_sum - bb_diag

        b_a = (torch.mul(b[0], b[1]) + bu + ub) / ((1 - K).view(-1, 1).expand(b[0].shape))
        u_a = torch.mul(u[0], u[1]) / ((1 - K).view(-1, 1).expand(u[0].shape))

        S_a = self.args.n_classes / u_a
        e_a = torch.mul(b_a, S_a.expand(b_a.shape))
        alpha_a = e_a + 1

        return alpha_a

    def forward(self, ecape, re, fb, mf, soec, A, B, all):
        """
        Forward pass of TMC model

        Args:
            ecape: ECAPE view features
            re: RE view features  
            fb: FB view features
            mf: MF view features
            soec: SOEC view features
            A: A view features
            B: B view features
            all: Combined/global features

        Returns:
            Tuple of alpha values for each view and combined result
        """
        # Process each view through its classifier
        ecape_out = ecape
        for layer in self.clf_ecape:
            ecape_out = layer(ecape_out)

        re_out = re
        for layer in self.clf_re:
            re_out = layer(re_out)

        fb_out = fb
        for layer in self.clf_fb:
            fb_out = layer(fb_out)

        mf_out = mf
        for layer in self.clf_mf:
            mf_out = layer(mf_out)

        soec_out = soec
        for layer in self.clf_soec:
            soec_out = layer(soec_out)

        A_out = A
        for layer in self.clf_A:
            A_out = layer(A_out)

        B_out = B
        for layer in self.clf_B:
            B_out = layer(B_out)

        # Convert logits to evidence using softplus activation
        ecape_evidence = F.softplus(ecape_out)
        re_evidence = F.softplus(re_out)
        fb_evidence = F.softplus(fb_out)
        mf_evidence = F.softplus(mf)  # Note: Using raw mf instead of mf_out
        soec_evidence = F.softplus(soec_out)
        A_evidence = F.softplus(A_out)
        B_evidence = F.softplus(B_out)

        # Convert evidence to Dirichlet parameters (alpha)
        ecape_alpha = ecape_evidence + 1
        re_alpha = re_evidence + 1
        fb_alpha = fb_evidence + 1
        mf_alpha = mf_evidence + 1
        soec_alpha = soec_evidence + 1
        A_alpha = A_evidence + 1
        B_alpha = B_evidence + 1

        # Combine all evidences using Dempster-Shafer rule
        list_fusions = [ecape_alpha, re_alpha, fb_alpha, mf_alpha, soec_alpha, A_alpha, B_alpha]
        last_alpha = list_fusions[-1]

        # Iteratively combine evidences
        for x in list_fusions[:-1]:
            last_alpha = self.DS_Combin_two(last_alpha, x)

        depth_rgb_alpha = last_alpha

        return ecape_alpha, re_alpha, fb_alpha, mf_alpha, soec_alpha, A_alpha, B_alpha, depth_rgb_alpha


# ======================== Utility Functions ========================
def sign_sqrt(x):
    """
    Compute signed square root

    Args:
        x: Input tensor

    Returns:
        Signed square root of x
    """
    eps = 1e-10
    return torch.sign(x) * torch.sqrt(torch.abs(x) + eps)


def l2_norm(x):
    """
    Compute L2 normalization

    Args:
        x: Input tensor

    Returns:
        L2 normalized tensor
    """
    return torch.nn.functional.normalize(x, p=2, dim=-1)


# ======================== ETMC Model ========================
class ETEF(TMC):
    """
    Extended Trusted Multi-view Classification Model
    Extends TMC with enhanced feature fusion capabilities
    """

    def __init__(self, args):
        """
        Initialize ETMC model

        Args:
            args: Arguments containing model configuration
        """
        super(ETEF, self).__init__(args)

        # Feature fusion configuration
        last_size = 128  # Fixed size for fusion layer
        self.fusion_x = None
        self.clf = nn.ModuleList()
        self.nb_feats = args.dims  # Use dims from args instead of global nb_feats
        self.view_models = nn.ModuleList()

        # Build fusion classifier
        for hidden in args.hidden:
            self.clf.append(nn.Linear(last_size, hidden))
            self.clf.append(nn.BatchNorm1d(hidden))
            self.clf.append(nn.ReLU())
            # self.clf.append(nn.Dropout(args.dropout))
            last_size = hidden
        self.clf.append(nn.Linear(last_size, args.n_classes))

        # Build view-specific preprocessing models
        # Adjust the range based on actual number of views
        for i in range(len(self.nb_feats)):
            self.view_models.append(nn.Sequential(
                nn.BatchNorm1d(self.nb_feats[i]),
                nn.Linear(self.nb_feats[i], FUSED_NB_FEATS),
                # nn.BatchNorm1d(FUSED_NB_FEATS),
                nn.ReLU()
            ))

    def fusion(self, x1, x2, fused_nb_feats, way='add'):
        """
        Fuse two feature tensors using specified method

        Args:
            x1: First feature tensor
            x2: Second feature tensor
            fused_nb_feats: Target dimension for fused features
            way: Fusion method ('add', 'mul', 'cat', 'max', 'avg')

        Returns:
            Fused feature tensor
        """
        if way == 'add':
            self.fusion_x = torch.add(x1, x2)
        elif way == 'mul':
            self.fusion_x = torch.mul(x1, x2)
        elif way == 'cat':
            self.fusion_x = torch.cat((x1, x2), dim=-1)
            device_index = self.fusion_x.device.index if self.fusion_x.is_cuda else 'cpu'
            device = f'cuda:{device_index}' if device_index != 'cpu' else 'cpu'
            # Create linear layer to reduce concatenated features to target dimension
            linear_layer = torch.nn.Linear(self.fusion_x.shape[1], fused_nb_feats).to(device)
            self.fusion_x = linear_layer(self.fusion_x)
        elif way == 'max':
            self.fusion_x = torch.max(x1, x2)
        elif way == 'avg':
            self.fusion_x = torch.div(torch.add(x1, x2), 2.0)
        return self.fusion_x

    def forward(self, ecape, re, fb, mf, soec, A, B, all):
        """
        Forward pass of ETMC model with enhanced fusion

        Args:
            ecape: ECAPE view features
            re: RE view features
            fb: FB view features
            mf: MF view features
            soec: SOEC view features
            A: A view features
            B: B view features
            all: Combined/global features

        Returns:
            Tuple of alpha values for each view, pseudo view, and combined result
        """
        # Process each view through its classifier, saving intermediate features for fusion
        ecape_out = ecape
        ecape_fusion = ecape
        for i, layer in enumerate(self.clf_ecape):
            cnt = len(self.clf_ecape)
            ecape_out = layer(ecape_out)
            if i == cnt - 2:  # Save features before final classification layer
                ecape_fusion = ecape_out

        re_out = re
        re_fusion = re
        for i, layer in enumerate(self.clf_re):
            cnt = len(self.clf_re)
            re_out = layer(re_out)
            if i == cnt - 2:
                re_fusion = re_out

        fb_out = fb
        fb_fusion = fb
        for i, layer in enumerate(self.clf_fb):
            fb_out = layer(fb_out)
            cnt = len(self.clf_fb)
            if i == cnt - 2:
                fb_fusion = fb_out

        mf_out = mf
        mf_fusion = mf_out
        for i, layer in enumerate(self.clf_mf):
            mf_out = layer(mf_out)
            cnt = len(self.clf_mf)
            if i == cnt - 2:
                mf_fusion = mf_out

        soec_out = soec
        soec_fusion = soec
        for i, layer in enumerate(self.clf_soec):
            soec_out = layer(soec_out)
            cnt = len(self.clf_soec)
            if i == cnt - 2:
                soec_fusion = soec_out

        A_out = A
        A_fusion = A
        for i, layer in enumerate(self.clf_A):
            A_out = layer(A_out)
            cnt = len(self.clf_A)
            if i == cnt - 2:
                A_fusion = A_out

        B_out = B
        B_fusion = B
        for i, layer in enumerate(self.clf_B):
            B_out = layer(B_out)
            cnt = len(self.clf_B)
            if i == cnt - 2:
                B_fusion = B_out

        # Parse fusion configuration
        fusion_list = FUSION_LIST
        fusion_xx = [ecape_fusion, re_fusion, fb_fusion, mf_fusion, soec_fusion, A_fusion, B_fusion]

        my_fusion_list = [int(x) for x in fusion_list.split('-')]
        n = len(my_fusion_list)
        view_cnt = int(n / 2 + 1)
        fusion_num = my_fusion_list[:view_cnt]
        opt_num = my_fusion_list[view_cnt:]

        # Concatenate each view with global features
        ecape_1 = torch.cat([ecape, all], -1)
        re_1 = torch.cat([re, all], -1)
        fb_1 = torch.cat([fb, all], -1)
        mf_1 = torch.cat([mf, all], -1)
        soec_1 = torch.cat([soec, all], -1)
        A_1 = torch.cat([A, all], -1)
        B_1 = torch.cat([B, all], -1)

        # Prepare features for fusion
        init_features = [ecape_1, re_1, fb_1, mf_1, soec_1, A_1, B_1]
        # Alternative: use raw features without concatenation
        # init_features = [ecape, re, fb, mf, soec, A, B]

        # Select views based on fusion configuration
        views = [init_features[i] for i in fusion_num]

        # Process each selected view through its preprocessing model
        x_processed = []
        for i in range(len(views)):
            if i < len(self.view_models):
                x_processed.append(self.view_models[i](views[i]))

        # Perform iterative fusion
        if x_processed:
            self.fusion_x = x_processed[0]
            x_processed = x_processed[1:]
            fusion_num = fusion_num[1:]

            for i, view in enumerate(fusion_num):
                if i < len(x_processed) and i < len(opt_num):
                    fusion_method = FUSION_METHODS[opt_num[i]] if opt_num[i] < len(FUSION_METHODS) else 'add'
                    self.fusion_x = self.fusion(self.fusion_x, x_processed[i], FUSED_NB_FEATS, fusion_method)

        # Process fused features through classifier
        pseudo_out = self.fusion_x
        for i, layer in enumerate(self.clf):
            pseudo_out = layer(pseudo_out)

        # Convert outputs to evidence using softplus
        ecape_evidence = F.softplus(ecape_out)
        re_evidence = F.softplus(re_out)
        fb_evidence = F.softplus(fb_out)
        mf_evidence = F.softplus(mf_out)
        soec_evidence = F.softplus(soec_out)
        A_evidence = F.softplus(A_out)
        B_evidence = F.softplus(B_out)
        pseudo_evidence = F.softplus(pseudo_out)

        # Convert evidence to Dirichlet parameters
        ecape_alpha = ecape_evidence + 1
        re_alpha = re_evidence + 1
        fb_alpha = fb_evidence + 1
        mf_alpha = mf_evidence + 1
        soec_alpha = soec_evidence + 1
        A_alpha = A_evidence + 1
        B_alpha = B_evidence + 1
        pseudo_alpha = pseudo_evidence + 1

        # Combine all evidences using Dempster-Shafer rule
        list_fusions = [ecape_alpha, re_alpha, fb_alpha, mf_alpha, soec_alpha,
                        A_alpha, B_alpha, pseudo_alpha]


        # Iteratively combine all evidences
        last_alpha = list_fusions[-1]
        for x in list_fusions:
            last_alpha = self.DS_Combin_two(last_alpha, x)

        depth_rgb_alpha = last_alpha

        return (ecape_alpha, re_alpha, fb_alpha, mf_alpha, soec_alpha,
                A_alpha, B_alpha, pseudo_alpha, depth_rgb_alpha)