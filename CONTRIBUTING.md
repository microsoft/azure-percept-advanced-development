# Contributing

## Intro

This project welcomes contributions and suggestions. Most contributions require you to
agree to a Contributor License Agreement (CLA) declaring that you have the right to,
and actually do, grant us the rights to use your contribution. For details, visit
https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need
to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the
instructions provided by the bot. You will only need to do this once across all repositories using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Expectations and Types of Contributions

Please note that while we have open-sourced this repository to enable experimentation with the Azure Percept device,
it should be considered to be in **preview/beta**, and therefore subject to changes without notice. Things that
you contribute will be helpful no matter what but the following are things that are especially helpful
and are likely to be accepted:

* Updates to the README
* Tidying up of the code base
* Bug fixes (with repro steps for the bug that show how you have fixed it)
* Adding tests
* New model support

In particular, adding new model support is likely to survive changes to the code as we move through public preview and onto
general availability.

We welcome all contributions with adequite testing and justification, but please keep in mind that:

* large architectural changes
* large refactoring work
* adding new libraries/dependencies
* adding new features

will likely require quite a bit more review and may ultimately be rejected. If you want to help out, please
communicate with us by opening a GitHub issue with your proposed change and justification for it **before** actually
implementing it.

## Coding Conventions

We have tried to follow some basic coding conventions just to keep things looking neat and clean.
They have never been officially enforced (why be dogmatic?), but please do try to follow them:

* Line length doesn't really matter these days. Don't split a long line up just because it hits
  some arbitrary line length. Split it up if it *feels* like it should be split up.
* Macros should be `ALL_CAPS_WITH_UNDERSCORES`. Same with module-level constants (or constants
  exposed in header files).
* Variables should be `snake_case_not_camel_case` and should be as descriptive as necessary.
  Please don't name your variables `a` and `b`, unless you are coding from an equation in a
  paper, in which case please use comments to explain.
* Function names should also be snake case.
* Class names and struct names should be `CamelCase` and should be short and sweet.
* Don't use `using namespace std` or anything similar. A long explicit name is better than
  a short confusing one. Yes, you have to type more, but you read the code much more often
  than you type it. Besides, most editors will do a decent job code-completing for you anyway.
* Comments should be used anytime you think something might be confusing. Having said that,
  try to make it as not-confusing as possible first.
* Every function, macro, constant, typedef, class, struct, whathaveyou should be commented using
  Doxygen-ish style comments. We don't use Doxygen, but it makes for a reasonable standard.
* We put our curly braces on their own lines for the most part, though there are esoteric
  places where we don't. Again, we're not dogmatic.

There is ample code here to examine for style examples. Just take a look around and try to make
your code match the style of what's already here.
