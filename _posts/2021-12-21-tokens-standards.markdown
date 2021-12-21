---
title: ERC-20 vs ERC-721 vs ERC-777 vs ERC-1155
date: 2021-12-21
tags: [Web3, Tokens, NFT]
---

<h2>Token Standards</h2>

The token standards are used for creating smart contracts on the Ethereum blockchain. These standards enforces a set of rules and implementation of some basic functions in the smart contracts.

Key differences between the most popular token standards built on Ethereum.

# ERC-20

ERC-20 (Etheruem for Request Comments) are fungible tokens which means each of them are identical and can be exchanged with one another. 

Many popular projects such as [Chainlink](http://chain.link), USD Coin uses the ERC-20 standard.

# ERC-721

ERC-721 are non-fungible tokens which means each of them are unique in nature. 

These tokens are heavily used in platforms offering unique collectibles such as CryptoKitties, Bored Ape Yacht Club etc. Since these tokens are one of a kind, they have an intrinsic value associated with them.

# ERC-777

ERC-777 is an improvement over ERC-20 standard.

This standard provides the capability of hooks for the smart contracts. Hooks (similar to Webhooks) gets executed whenever the tokens are sent/received by the smart contract.

# ERC-1155

ERC-1155 is a multi token standard which includes combination of fungible tokens, non-fungible tokens using a single smart contract. It is a mix of ERC-20 and ERC-721 standard.

<br />

Technically, NFTs (Non Fungible Tokens) are truly unique tokens which means they should not have more than 1 supply, each token is one of a kind.

ERC-1155 is not a "truly" NFT since you can create more than 1 copies of same token.