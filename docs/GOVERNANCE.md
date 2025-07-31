# AI Scientist v2 - Project Governance

This document outlines the governance structure, decision-making processes, and contribution guidelines for the AI Scientist v2 project.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Roles and Responsibilities](#roles-and-responsibilities)  
3. [Decision Making Process](#decision-making-process)
4. [Release Management](#release-management)
5. [Security and Compliance](#security-and-compliance)
6. [Community Guidelines](#community-guidelines)
7. [Conflict Resolution](#conflict-resolution)

## Project Structure

### Organization Hierarchy

```
AI Scientist v2 Project
├── Steering Committee
├── Core Maintainers Team
├── Special Interest Groups (SIGs)
│   ├── AI/ML Research SIG
│   ├── Security SIG
│   ├── DevOps SIG
│   ├── Documentation SIG
│   └── Community SIG
└── Contributors
```

### Repository Organization

The project follows a multi-repository structure under the Terragon Labs organization:

- **Main Repository**: `SakanaAI/AI-Scientist-v2` - Core AI Scientist implementation
- **Documentation**: Comprehensive docs in `/docs` directory
- **Examples**: Research examples and templates in `/ai_scientist/fewshot_examples`
- **Templates**: LaTeX and workflow templates

## Roles and Responsibilities

### Steering Committee

**Members**: 3-5 senior technical leaders
**Term**: 2 years, renewable
**Responsibilities**:
- Set overall project direction and strategy
- Approve major architectural changes
- Resolve escalated conflicts
- Manage project resources and funding decisions
- Approve new core maintainers

**Current Members**:
- Project Lead: Technical vision and strategy
- Research Director: AI/ML research direction  
- Engineering Director: Technical architecture
- Security Lead: Security strategy and compliance
- Community Lead: Community growth and engagement

### Core Maintainers Team

**Selection Criteria**:
- 6+ months of consistent contributions
- Deep understanding of project architecture
- Demonstrated leadership in technical discussions
- Commitment to project values and code of conduct

**Responsibilities**:
- Review and merge pull requests
- Triage and prioritize issues
- Mentor new contributors
- Maintain code quality and architectural consistency
- Lead technical design discussions
- Participate in release planning

**Current Teams**:
- `@terragon-labs/core-maintainers` - Overall project maintenance
- `@terragon-labs/ai-ml-team` - AI/ML research and algorithms
- `@terragon-labs/security-team` - Security and compliance
- `@terragon-labs/devops-team` - Infrastructure and deployment

### Special Interest Groups (SIGs)

#### AI/ML Research SIG
**Focus**: Research methodology, algorithm development, model integration
**Lead**: Research Director
**Members**: AI/ML researchers, algorithm specialists
**Meetings**: Bi-weekly technical reviews

#### Security SIG  
**Focus**: Security architecture, vulnerability management, compliance
**Lead**: Security Lead
**Members**: Security engineers, compliance specialists
**Meetings**: Weekly security reviews, monthly threat modeling

#### DevOps SIG
**Focus**: Infrastructure, CI/CD, deployment automation
**Lead**: DevOps Lead
**Members**: DevOps engineers, SRE specialists
**Meetings**: Weekly infrastructure reviews

#### Documentation SIG
**Focus**: Documentation quality, user experience, tutorials
**Lead**: Documentation Lead  
**Members**: Technical writers, UX specialists
**Meetings**: Monthly documentation reviews

#### Community SIG
**Focus**: Community growth, contributor onboarding, events
**Lead**: Community Lead
**Members**: Community managers, developer advocates
**Meetings**: Monthly community reviews

## Decision Making Process

### Decision Categories

#### Category 1: Major Decisions
**Examples**: Architecture changes, new major features, security policies
**Process**: 
1. RFC (Request for Comments) process
2. Public discussion period (2 weeks minimum)
3. Core maintainer review and vote
4. Steering committee approval (if needed)
5. Implementation planning

#### Category 2: Technical Decisions
**Examples**: API changes, dependency updates, refactoring
**Process**:
1. Technical proposal in GitHub issue
2. Discussion with relevant SIG
3. Core maintainer consensus (lazy consensus model)
4. Implementation

#### Category 3: Operational Decisions
**Examples**: Bug fixes, documentation updates, minor features
**Process**:
1. Standard pull request workflow
2. Code review by appropriate team members
3. Merge after approval

### RFC Process

For major decisions, we use a formal RFC (Request for Comments) process:

1. **Proposal**: Create RFC document in `/docs/rfcs/`
2. **Discussion**: Open GitHub issue for community input
3. **Review**: SIG and core maintainer review
4. **Decision**: Formal acceptance/rejection with rationale
5. **Implementation**: Tracking issue with milestones

**RFC Template**: [RFC Template](./rfcs/RFC-TEMPLATE.md)

### Consensus Building

- **Lazy Consensus**: Assumes agreement unless objections are raised
- **Active Consensus**: Requires explicit approval from stakeholders
- **Majority Vote**: Used when consensus cannot be reached
- **Veto Power**: Core maintainers can veto decisions affecting their domain

## Release Management

### Release Cycle

- **Major Releases** (X.0.0): Every 6 months, includes breaking changes
- **Minor Releases** (X.Y.0): Monthly, new features and improvements  
- **Patch Releases** (X.Y.Z): As needed, bug fixes and security updates

### Release Process

1. **Planning Phase** (2 weeks before release)
   - Feature freeze for major/minor releases
   - Release candidate preparation
   - Testing and validation

2. **Release Candidate Phase** (1 week)
   - RC testing by core maintainers
   - Community testing and feedback
   - Critical bug fixes only

3. **Release Phase**
   - Final testing and sign-off
   - Release creation and artifact publishing
   - Documentation updates
   - Community announcement

### Release Roles

- **Release Manager**: Coordinates release process (rotating role)
- **Quality Lead**: Oversees testing and validation
- **Security Lead**: Reviews security implications
- **Documentation Lead**: Ensures documentation is current

### Version Compatibility

- **API Compatibility**: Semantic versioning for public APIs
- **Configuration Compatibility**: Migration guides for breaking changes
- **Model Compatibility**: Backward compatibility for research models
- **Data Compatibility**: Clear upgrade paths for data formats

## Security and Compliance

### Security Governance

- **Security Team**: Dedicated security specialists
- **Security Reviews**: Required for all security-related changes
- **Threat Modeling**: Regular security architecture reviews
- **Incident Response**: Defined process for security incidents

### Vulnerability Management

1. **Reporting**: Security issues reported to security@terragonlabs.ai
2. **Triage**: Security team assessment within 24 hours
3. **Response**: Fix development and testing
4. **Disclosure**: Coordinated disclosure with advisory publication
5. **Follow-up**: Post-incident review and process improvement

### Compliance Framework

- **Industry Standards**: Compliance with relevant AI/ML industry standards
- **Data Privacy**: GDPR and other privacy regulation compliance
- **Export Control**: Review of export control implications
- **License Compliance**: Regular license compatibility audits

## Community Guidelines

### Code of Conduct

All community members must follow our [Code of Conduct](../CODE_OF_CONDUCT.md), which emphasizes:

- Respectful and inclusive communication
- Professional behavior in all interactions
- Zero tolerance for harassment or discrimination
- Commitment to learning and helping others

### Contribution Process

1. **First-Time Contributors**:
   - Read contributing guidelines
   - Start with "good first issue" labels
   - Join community discussions
   - Attend newcomer meetings

2. **Regular Contributors**:
   - Participate in SIG meetings
   - Take on mentor roles
   - Lead feature development
   - Contribute to documentation

3. **Potential Maintainers**:
   - Demonstrate consistent contributions
   - Show technical leadership
   - Participate in project governance
   - Commitment to project values

### Recognition Program

- **Contributor Highlights**: Monthly recognition in newsletters
- **Conference Speaking**: Opportunities to present project work
- **Mentorship Program**: Pairing experienced contributors with newcomers
- **Swag and Rewards**: Project merchandise for significant contributions

## Conflict Resolution

### Resolution Process

1. **Direct Discussion**: Encourage direct communication between parties
2. **Mediation**: SIG leads or core maintainers facilitate discussion
3. **Escalation**: Steering committee review for unresolved conflicts
4. **Final Decision**: Steering committee binding decision
5. **Appeal Process**: Limited appeal rights to project leadership

### Types of Conflicts

#### Technical Disagreements
- **Process**: Technical discussion in GitHub issues/PRs
- **Resolution**: Core maintainer consensus or architectural review
- **Appeal**: Steering committee technical review

#### Behavioral Issues  
- **Process**: Private discussion with community leads
- **Resolution**: Coaching, warnings, or temporary restrictions
- **Appeal**: Independent review by steering committee

#### Governance Disputes
- **Process**: Formal governance review process
- **Resolution**: Policy clarification or modification
- **Appeal**: Community feedback and steering committee decision

### Enforcement Mechanisms

- **Warnings**: For minor code of conduct violations
- **Temporary Restrictions**: Limited repository access or participation
- **Permanent Ban**: For severe or repeated violations
- **Appeal Rights**: All enforcement actions can be appealed

## Governance Evolution

### Annual Review

The governance structure undergoes annual review including:

- Effectiveness assessment of current structure
- Community feedback collection
- Process improvement identification
- Structural adjustments as needed

### Amendment Process

Governance changes require:

1. **Proposal**: RFC for governance changes
2. **Community Input**: 4-week public comment period
3. **Review**: Steering committee and SIG lead review
4. **Approval**: Steering committee supermajority (4/5) vote
5. **Implementation**: Gradual rollout with feedback collection

### Metrics and KPIs

We track governance effectiveness through:

- **Contributor Growth**: New contributor onboarding rates
- **Decision Speed**: Time from proposal to decision
- **Community Health**: Satisfaction surveys and engagement metrics
- **Code Quality**: Technical debt and security metrics
- **Project Velocity**: Feature delivery and release frequency

## Communication Channels

### Official Channels

- **GitHub**: Primary development and project management
- **Documentation**: Comprehensive docs at docs.terragonlabs.ai
- **Mailing Lists**: Major announcements and governance discussions
- **Slack/Discord**: Real-time community communication
- **Blog**: Regular project updates and technical posts

### Meeting Schedule

- **Steering Committee**: Monthly strategic meetings
- **Core Maintainers**: Weekly technical sync
- **SIG Meetings**: Bi-weekly domain-specific discussions  
- **Community Meetings**: Monthly open community calls
- **Office Hours**: Weekly open Q&A sessions

## Related Resources

- [Contributing Guidelines](../CONTRIBUTING.md)
- [Code of Conduct](../CODE_OF_CONDUCT.md)
- [Security Policy](../SECURITY.md)
- [Technical Architecture](../ARCHITECTURE.md)
- [Branch Protection Guide](./BRANCH_PROTECTION_GUIDE.md)
- [Release Process](./workflows/RELEASE_AUTOMATION.md)