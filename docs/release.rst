.. _release:

Development Processes and Tools
===============================

Versioning
----------

pvcaptest follows `semantic versioning <https://semver.org/>`__.

pvcaptest uses `setuptools-scm <https://github.com/pypa/setuptools_scm>`__ to automatically derive the version from git tags. The version is dynamically generated at build time and reflects the current git state.

Version format:

- Tagged commit ``v0.9.0`` → version ``0.9.0``
- After tag, uncommitted changes → ``0.9.0.post0.dev1+dirty``
- 3 commits after tag → ``0.9.0.post0.dev3+g1234abc``

Access the version in code::

    import captest
    print(captest.__version__)  # Uses importlib.metadata

Git Workflow
------------

The pvcaptest project follows the git workflow / branching strategy as described by the following bullets.

- There is a single master branch from which feature branches originate.
- There are no development or release branches.
- Version tags are used to initiate testing and releases to pypi. So, any commit not tagged on master or any other branch is part of development.
- Releases should be made from an upstream repository under an organization account.
- Development work should occur on feature branch on a fork of the upstream organization repository under the contributor's github profile.

This approach is based on the strategy described by Mark Mikofski in his blog post, `Winning Workflow <https://poquitopicante.blogspot.com/2016/10/winning-workflow.html>`__.

Development Workflow with uv
-----------------------------

The project uses `uv <https://github.com/astral-sh/uv>`__ for dependency management and development workflows.

Install dependencies::

    uv sync                 # Install core dependencies
    uv sync --all-extras    # Install all optional dependencies

Run commands in the environment::

    uv run python script.py
    uv run pytest

Add/remove dependencies::

    uv add package-name
    uv remove package-name

Build distributions::

    uv build


Testing
-------

Github actions are used for continuous integration testing.  All pull requests are built and tested and version tags on upstream master are built and tested and then, if passing, released to PyPI.

Release Checklist
-----------------

**These steps may change as this process is implemented and tested.**

- Pull from upstream master to update your local clone of your fork
- Push to origin master to keep your local clone up up to date
- Checkout new feature branch
- Create test and pull request
- Update change log
- Complete changes in branch and ensure pull request passes tests

Maintainer will:

- checkout master branch
- merge release branch into master
- tag master branch (this will trigger versioneer to update version)
- push updates and tag to github
- delete the release branch

Pushing a tag beginning with v from master in pvcaptest/pvcaptest will trigger the Github action publish, which will build and publish to PyPI.

``git tag -a v0.X.X -m 'v0.X.X'``

``git push origin v0.X.X``
