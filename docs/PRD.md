# RAG Factory — Product Requirements Document

## Vision
A production-grade, cloud-portable RAG platform that stamps out retrieval-augmented generation systems from a catalog of proven patterns — from naive single-retriever setups to fully agentic multi-hop reasoning pipelines.

## Problem
Organizations building RAG systems face:
1. **Pattern paralysis** — too many retrieval strategies, no guidance on which to use
2. **Infrastructure sprawl** — vector stores, LLMs, caches, and pipelines that don't compose
3. **Cloud lock-in** — tightly coupled to one cloud's managed services
4. **Production gap** — demos work but production deployments fail on security, multi-tenancy, and scale

## Solution
RAG Factory provides:
- **5 battle-tested blueprints** (naive, hybrid, graph, multimodal, agentic)
- **4-Ops module system** (NetOps, SecOps, AppOps, DevOps) with cloud-free contracts
- **Factory stamping** — `scaffold.sh` stamps any blueprint into any environment
- **Reference application** — fully implemented agentic RAG with strict typing and tests
- **Policy as code** — OPA and Checkov guardrails baked in

## Target Users
1. **Cloud architects** choosing and implementing RAG patterns
2. **ML/AI engineers** building production retrieval pipelines
3. **Platform teams** standardizing RAG infrastructure across teams
4. **Citadel students** learning enterprise RAG architecture

## Success Metrics
- Time from zero to production RAG: < 4 hours
- Blueprint portability: same blueprint deploys on 3 clouds
- Search quality: NDCG@10 > 0.7 on standard benchmarks
- Test coverage: > 80% on search subsystem
