import torch
import torch.nn as nn


from scripts.kg_objects import Entity, Relation
from scripts.utils import DataFromJSON
from scripts.TransformerModified import TransformerEncoderRelationBias

class TripletSymmetricVAE(nn.Module):
    def __init__(self, model_conf: DataFromJSON, gpu_device: torch.device = None, dicts_set: dict = None):
        super(TripletSymmetricVAE, self).__init__()
        self.n_embed = model_conf.n_embed
        self.n_hidden = model_conf.n_hidden
        self.n_latent = model_conf.n_latent
        self.epochs = model_conf.epochs
        self.gpu_device = gpu_device
        self.dicts_set = dicts_set
        self.vocab_size = len(dicts_set["links_states"])

        self.build_model(self.n_embed, self.n_hidden, self.n_latent, self.vocab_size)

    def build_model(self, n_embed: int, n_hidden: tuple[int], n_latent: int, vocab_size: int):
        """
        Build the embedding, encoder, decoder, and projector networks.
        """
        # Embedding layer
        self.embed = nn.Embedding(vocab_size, n_embed)

        # Encoder
        encoder_layers = []
        n_hidden_enc = [n_embed] + list(n_hidden)
        for i in range(len(n_hidden_enc) - 1):
            encoder_layers.append(nn.Linear(n_hidden_enc[i], n_hidden_enc[i + 1]))
            encoder_layers.append(nn.ReLU())
        encoder_layers.append(nn.Linear(n_hidden_enc[-1], n_latent * 2))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        n_hidden_dec = [n_latent] + list(n_hidden)[::-1] + [n_embed]
        for i in range(len(n_hidden_dec) - 1):
            decoder_layers.append(nn.Linear(n_hidden_dec[i], n_hidden_dec[i + 1]))
            decoder_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*decoder_layers)

        # Projector
        proj_hidden = int((n_embed + vocab_size) / 2)
        proj_hidden = max(proj_hidden, 2*n_embed)
        self.projector = nn.Sequential(
            nn.Linear(n_latent, proj_hidden),
            nn.Linear(proj_hidden, vocab_size)
        )

        # Softmax layer for probability calculation
        self.softmax = nn.Softmax(dim=-1)

        # Initialize the weights
        self.init_weights()

    def init_weights(self):
        """
        Initialize the weights of the model.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, idx, noise_factor: float = 1.0):
        """
        Forward pass through the model.
        """
        x = self.embed(idx)
        mu, logvar = self.encoder(x).chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar, noise_factor)
        x_hat = self.decoder(z)
        logits = self.projector(z)
        probs = self.softmax(logits)
        return x, x_hat, logits, probs, mu, logvar

    def reparameterize(self, mu, logvar, noise_factor: float = 1.0):
        """
        Reparameterization trick for sampling from a Gaussian.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) * noise_factor
        return mu + eps * std
    
    def train_model(self):
        """
        Train the link prediction model.
        """
        print("Training the link prediction model...")

        # Set the model to training mode
        self.train()

        # Define the optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        # Define the loss function
        loss_cre = torch.nn.CrossEntropyLoss().to(self.gpu_device)
        loss_mse = torch.nn.MSELoss().to(self.gpu_device)

        # Get the links states with the value equals to 1
        value_1_links_states = {key: value for key, value in self.dicts["links_states"].items() if value == 1}

        # Define the training loop
        for epoch in range(self.epochs):
            # Generate random samples
            input = torch.tensor([self.dicts["link_to_int"][key] for key in value_1_links_states.keys()])

            # Randomly shuffle the indices
            input = input[torch.randperm(input.size(0))].to(self.gpu_device)

            # Forward pass
            x, x_hat, logits, probs, mu, logvar = self(input)

            # Get the values which should be 1
            iden = torch.eye(self.vocab_size).to(self.gpu_device)
            exp_probs = iden[input, :]

            # Compute the loss
            embed_loss = loss_mse(x, x_hat) # mse or cre?
            probs_loss = loss_cre(logits, exp_probs)
            loss: torch.Tensor = embed_loss + probs_loss

            # Print the loss
            print(f"Epoch: {epoch}, Total Loss: {loss.item():.5f}, Embedding Loss: {embed_loss.item():.5f}, Probs Loss: {probs_loss.item():.5f}")

            # Zero the gradients
            optimizer.zero_grad()

            # Backward pass
            loss.backward()

            # Optimize
            optimizer.step()

        return self
    
    def inference_model(self, noise_factor: float = 1.0):
        """
        Evaluate the link prediction model.
        """
        print("Evaluating the link prediction model...")

        # Set the model to evaluation mode
        self.eval()

        # Get the links states with the value equals to 1
        value_1_links_states = {key: value for key, value in self.dicts["links_states"].items() if value == 1}

        # Obtain inputs
        input = torch.tensor([self.dicts["link_to_int"][key] for key in value_1_links_states.keys()])

        for i in range(input.size(0)):
            # Forward pass
            x, x_hat, logits, probs, mu, logvar = self(input[i], noise_factor=noise_factor)

            # Obtain top 3 predictions
            top_3 = torch.topk(probs, 3)

            # Print the top 3 predictions values and indices
            print("Top 3 predictions values:", [top_3.values[i].item() for i in range(top_3.values.size(0))])
            print("Top 3 predictions indices:", [top_3.indices[i].item() for i in range(top_3.indices.size(0))])

        return self
    
class PartRotSymmetricVAE(nn.Module):
    def __init__(self, model_conf: DataFromJSON, gpu_device: torch.device = None, dicts_set: dict = None):
        super(PartRotSymmetricVAE, self).__init__()
        self.n_embed_sect = model_conf.n_embed_sect
        self.n_hidden = model_conf.n_hidden
        self.n_latent = model_conf.n_latent
        self.epochs = model_conf.epochs
        self.gpu_device = gpu_device
        self.dicts_set = dicts_set
        self.entity_vocab_size = len(dicts_set["nodes_dict"])
        self.relation_vocab_size = len(dicts_set["relationships_dict"])

        self.build_model(self.n_embed_sect, self.n_hidden, self.n_latent, self.entity_vocab_size, self.relation_vocab_size)

    def build_model(self, n_embed_sect: int, n_hidden: tuple[int], n_latent: int, entity_vocab_size: int, relation_vocab_size: int):
        """
        Build the embedding, encoder, decoder, and projector networks.
        """
        # Embedding layer
        self.entity_embed = nn.Embedding(entity_vocab_size, n_embed_sect)
        self.relation_embed = nn.Embedding(relation_vocab_size, n_embed_sect)
        n_embed = n_embed_sect * 3

        # Encoder
        encoder_layers = []
        n_hidden_enc = [n_embed] + list(n_hidden)
        for i in range(len(n_hidden_enc) - 1):
            encoder_layers.append(nn.Linear(n_hidden_enc[i], n_hidden_enc[i + 1]))
            encoder_layers.append(nn.ReLU())
        encoder_layers.append(nn.Linear(n_hidden_enc[-1], n_latent * 2))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        n_hidden_dec = [n_latent] + list(n_hidden)[::-1] + [n_embed]
        for i in range(len(n_hidden_dec) - 1):
            decoder_layers.append(nn.Linear(n_hidden_dec[i], n_hidden_dec[i + 1]))
            decoder_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*decoder_layers)

        # Projector of the first entity
        proj_hidden = int((n_embed + entity_vocab_size) / 2)
        proj_hidden = max(proj_hidden, 2*n_embed)
        self.projector_ent1 = nn.Sequential(
            nn.Linear(n_latent, proj_hidden),
            nn.Linear(proj_hidden, entity_vocab_size)
        )

        # Projector of the relation
        proj_hidden = int((n_embed + relation_vocab_size) / 2)
        proj_hidden = max(proj_hidden, 2*n_embed)
        self.projector_rel = nn.Sequential(
            nn.Linear(n_latent, proj_hidden),
            nn.Linear(proj_hidden, relation_vocab_size)
        )

        # Projector of the second entity
        proj_hidden = int((n_embed + entity_vocab_size) / 2)
        proj_hidden = max(proj_hidden, 2*n_embed)
        self.projector_ent2 = nn.Sequential(
            nn.Linear(n_latent, proj_hidden),
            nn.Linear(proj_hidden, entity_vocab_size)
        )

        # Softmax layer for probability calculation
        self.softmax = nn.Softmax(dim=-1)

        # Initialize the weights
        self.init_weights()

    def init_weights(self):
        """
        Initialize the weights of the model.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, idx_ent1, idx_rel, idx_ent2, noise_factor: float = 1.0):
        """
        Forward pass through the model.
        """
        x = torch.cat((self.entity_embed(idx_ent1), self.relation_embed(idx_rel), self.entity_embed(idx_ent2)), dim=-1)
        mu, logvar = self.encoder(x).chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar, noise_factor)
        x_hat = self.decoder(z)

        logits_ent1 = self.projector_ent1(z)
        probs_ent1 = self.softmax(logits_ent1)

        logits_rel = self.projector_rel(z)
        probs_rel = self.softmax(logits_rel)

        logits_ent2 = self.projector_ent2(z)
        probs_ent2 = self.softmax(logits_ent2)
        return x, x_hat, (logits_ent1, logits_rel, logits_ent2), (probs_ent1, probs_rel, probs_ent2), mu, logvar

    def reparameterize(self, mu, logvar, noise_factor: float = 1.0):
        """
        Reparameterization trick for sampling from a Gaussian.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) * noise_factor
        return mu + eps * std

    def train_model(self):
        """
        Train the link prediction model.
        """
        print("Training the link prediction model...")

        # Set the model to training mode
        self.train()

        # Define the optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        # Define the loss function
        loss_cre = torch.nn.CrossEntropyLoss().to(self.gpu_device)
        loss_mse = torch.nn.MSELoss().to(self.gpu_device)

        # Get the indices of the entities and relationships from links states with the value equals to 1
        triads = []
        for i, key, value in enumerate(self.dicts["links_states"].items()):
            value_1_links_states = {key: value}
            if value > 1e5:
                break
            
        value_1_links_states = {key: value for key, value in self.dicts["links_states"].items() if value == 1}
        for key, value in value_1_links_states.items():
            ent1_idx = torch.tensor(self.dicts["node_to_int"][key[0]])
            rel_idx = torch.tensor(self.dicts["relationship_to_int"][key[1]])
            ent2_idx = torch.tensor(self.dicts["node_to_int"][key[2]])
            triads = triads + [ent1_idx, rel_idx, ent2_idx]

        # Convert the triads to a tensor
        triads = torch.tensor(triads).to(self.gpu_device)

        # Define the training loop
        for epoch in range(self.epochs):
            # Randomly sample the triads
            input = triads[torch.randperm(triads.size(0))]

            # Forward pass
            x, x_hat, logits, probs, mu, logvar = self(input)
            ent1_logits, rel_logits, ent2_logits = logits

            # Get the values which should be 1
            iden = torch.eye(self.vocab_size).to(self.gpu_device)
            ent1_exp_probs = iden[input[:, 0], :]
            rel_exp_probs = iden[input[:, 1], :]
            ent2_exp_probs = iden[input[:, 2], :]

            # Compute the loss
            embed_loss = loss_mse(x, x_hat) # mse or cre?
            ent1_probs_loss = loss_cre(ent1_logits, ent1_exp_probs)
            rel_probs_loss = loss_cre(rel_logits, rel_exp_probs)
            ent2_probs_loss = loss_cre(ent2_logits, ent2_exp_probs)
            loss: torch.Tensor = embed_loss + ent1_probs_loss + rel_probs_loss + ent2_probs_loss

            # Print the loss
            print(f"Epoch: {epoch}, Total Loss: {loss.item():.5f}, Embedding Loss: {embed_loss.item():.5f}, Ent1 Probs Loss: {ent1_probs_loss.item():.5f}, Rel Probs Loss: {rel_probs_loss.item():.5f}, Ent2 Probs Loss: {ent2_probs_loss.item():.5f}")

            # Zero the gradients
            optimizer.zero_grad()

            # Backward pass
            loss.backward()

            # Optimize
            optimizer.step()

        return self
    
    def inference_model(self, noise_factor: float = 1.0):
        """
        Evaluate the link prediction model.
        """
        print("Evaluating the link prediction model...")

        # Set the model to evaluation mode
        self.eval()