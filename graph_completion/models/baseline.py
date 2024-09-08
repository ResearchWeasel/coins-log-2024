"""
Module containing the implementation of a baseline embedding model and its loss function.
"""

from typing import Optional, Tuple

import pandas as pd
import torch as pt
from torch import nn

from graph_completion.graphs.preprocess import QueryData, QueryEmbeddingData
from graph_completion.models.graph_embedders import GraphEmbedderHpars
from graph_completion.models.link_rankers import LinkRankerHpars
from graph_completion.models.loss_terms import EmbeddingLossHpars


class BaselineEmbedder(nn.Module):
    def __init__(self, num_nodes: int, num_relations: int,
                 embedder_hpars: dict, transe_model: Optional[dict] = None):
        """
        Construct the embedding model.
        """

        super().__init__()
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.embedder_hpars = GraphEmbedderHpars.from_dict(embedder_hpars)

        self.algorithm = self.embedder_hpars.algorithm
        self.embedding_dim = self.embedder_hpars.embedding_dim
        self.mlp_num_hidden_layers = self.embedder_hpars.mlp_num_hidden_layers
        self.mlp_hidden_dim = self.embedder_hpars.mlp_hidden_dim

        self.embeddings_complex = self.embedder_hpars.algorithm in ["complex", "rotate", "q2b", "betae"]
        self.embeddings_relation_specific = self.embedder_hpars.algorithm in ["mlp", "gatne"]
        self.embedder_sparse = self.embedder_hpars.algorithm in ["sacn", "kbgat"]

        with pt.no_grad():
            if self.embedder_sparse and transe_model is not None:
                self.embedder_hpars.initial_entity_embeddings = transe_model["entity_embeddings.weight"]
                self.embedder_hpars.initial_r_embeddings = transe_model["r_embeddings.weight"]
        self.embedder = self.embedder_hpars.make()

    def set_graph_data(self, edge_data: pd.DataFrame, device: str):
        if self.embedder_sparse:
            self.embedder.set_edge_data(edge_data, device)

    def clear_x_full(self):
        if self.embedder_sparse:
            self.embedder.x_full = None

    def forward(self, query_data: QueryData) -> Tuple[pt.Tensor, pt.Tensor]:

        # Node embedding
        self.embedder.graph_embed(query_data)

        # Final embedding
        query_emb, answer_emb = self.embedder(query_data)

        return query_emb, answer_emb

    def embed_supervised(self, query_data: QueryData) -> QueryEmbeddingData:
        query_emb, answer_emb = self(query_data)
        return QueryEmbeddingData(c_query_emb=pt.zeros_like(query_emb), c_answer_emb=pt.zeros_like(answer_emb),
                                  query_emb=query_emb, answer_emb=answer_emb,
                                  y=query_data.query_tree_batch["y"], sample=query_data.query_tree_batch["sample"])


class BaselineLinkPredictor(nn.Module):
    def __init__(self, link_ranker_hpars: dict):
        """
        Construct the link prediction model.
        """

        super().__init__()
        self.link_ranker_hpars = LinkRankerHpars.from_dict(link_ranker_hpars)
        self.algorithm = link_ranker_hpars["algorithm"]

        self.link_ranker = self.link_ranker_hpars.make()

    def forward(self, query_emb: pt.Tensor, answer_emb: pt.Tensor) -> pt.Tensor:
        # Query scoring
        y = self.link_ranker(query_emb, answer_emb)

        if self.algorithm != "sacn":
            y = pt.sigmoid(y)

        return y


class BaselineLoss(nn.Module):
    def __init__(self, embedding_loss_hpars: dict):
        """
        Construct the loss function for the embedding model.
        """

        super().__init__()
        self.embedding_loss_hpars = EmbeddingLossHpars.from_dict(embedding_loss_hpars)
        self.algorithm = embedding_loss_hpars["algorithm"]
        self.learnable_link_ranker = embedding_loss_hpars["algorithm"] in ["mlp", "sacn", "kbgat"]

        self.embedding_loss = self.embedding_loss_hpars.make()

    def forward(self, query_emb_batch: QueryEmbeddingData, link_ranker: BaselineLinkPredictor):
        (_, _), (_, _), (query, query_neg), (answer, answer_neg) = query_emb_batch.sample_split_embeddings()

        if self.learnable_link_ranker:
            loss = self.embedding_loss(query, query_neg, answer, answer_neg, link_ranker.link_ranker)
        else:
            loss = self.embedding_loss(query, query_neg, answer, answer_neg)

        return loss
