# Releasing

The package is published to PyPI as `gsdtrial` by
[`.github/workflows/release.yml`](.github/workflows/release.yml), using PyPI Trusted
Publishing (no API token is stored anywhere).

## One-time setup

1. **Repository name.** The workflow and the PyPI publishers below assume the repository is
   `rushadnihal/trialdesign`.
2. **GitHub environments.** In Settings > Environments, create `testpypi` and `pypi`. Give
   `pypi` a required reviewer (yourself), so the real upload waits for an explicit approval.
3. **TestPyPI publisher.** Sign in at <https://test.pypi.org>, open Your account > Publishing >
   Add a new pending publisher, and enter: PyPI project name `gsdtrial`, owner `rushadnihal`,
   repository `trialdesign`, workflow `release.yml`, environment `testpypi`.
4. **PyPI publisher.** The same at <https://pypi.org> (a separate account from TestPyPI), with
   environment `pypi`.

A pending publisher reserves nothing: the project `gsdtrial` is created by the first upload.

## Each release

1. Set `version` in `pyproject.toml`, move the `CHANGELOG.md` entry from "unreleased" to
   today's date, and merge to `main` with CI green.
2. Tag the merge commit and push the tag (or create a GitHub release with a new `vX.Y.Z` tag,
   which does the same):

   ```bash
   git checkout main && git pull
   git tag -a v0.2.0 -m "v0.2.0"
   git push origin v0.2.0
   ```

3. The workflow checks that the tag matches the version, runs the tests, builds the sdist and
   wheel, uploads them to TestPyPI, and installs `gsdtrial` back from TestPyPI in a clean
   environment to run the CLI.
4. If that passes, it waits for approval of the `pypi` environment. Approve it in the Actions
   run to upload to PyPI. **A version number can never be reused on PyPI**, even after
   deleting it, so check the TestPyPI page first.

To try the build and TestPyPI upload without releasing, run the workflow by hand from the
Actions tab; it stops before PyPI.
